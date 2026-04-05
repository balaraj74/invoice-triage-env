#!/usr/bin/env python3
"""inference.py — Baseline inference script for InvoiceTriageEnv.

Uses the OpenAI API client to run a model against the environment.
Reads credentials from environment variables:
  API_BASE_URL  — The API endpoint for the LLM
  MODEL_NAME    — The model identifier to use for inference
  HF_TOKEN      — Your Hugging Face / API key

Produces reproducible baseline scores (0.0–1.0) on all 6 tasks.
Runtime: < 5 minutes on vcpu=2, memory=8gb.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from invoice_triage_env.models import (
    ActionType,
    InvoiceAction,
    InvoiceCategory,
    InvoiceObservation,
    IssueType,
    Priority,
)
from invoice_triage_env.server.invoice_triage_environment import (
    InvoiceTriageEnvironment,
    REWARD_CORRECT_CATEGORY,
    REWARD_CORRECT_PRIORITY,
    REWARD_CORRECT_EXTRACTION,
    REWARD_CORRECT_ISSUE_FLAG,
    REWARD_PO_MATCH_CORRECT,
    REWARD_CORRECT_DECISION,
    REWARD_STEP_COST,
)
from invoice_triage_env.tasks import ALL_TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 20
MAX_LLM_RETRIES = 2

# Maps string → enum for safe parsing
_ACTION_TYPE_MAP = {e.value: e for e in ActionType}
_CATEGORY_MAP = {e.value: e for e in InvoiceCategory}
_PRIORITY_MAP = {e.value: e for e in Priority}
_ISSUE_MAP = {e.value: e for e in IssueType}

# Fallback action if LLM fails
FALLBACK_ACTION = '{"action_type": "approve", "reason": "Fallback — LLM unavailable"}'

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an elite accounts-payable invoice triage agent. You MUST follow
the exact strategy and formats below to score maximum accuracy.

## RESPONSE FORMAT
Respond with EXACTLY ONE JSON object per turn. No markdown, no explanation.
{"action_type": "...", ...payload fields...}

## ACTION TYPES & PAYLOAD

| action_type     | payload                                                                 |
|-----------------|-------------------------------------------------------------------------|
| categorize      | "category": one of supplies|travel|software|consulting|utilities|maintenance|marketing|equipment|other |
| set_priority    | "priority": one of low|medium|high|urgent                               |
| extract_field   | "field_name": string, "field_value": string                             |
| validate_match  | "match_result": true or false                                           |
| flag_issue      | "issue_type": string, "issue_description": string                       |
| approve         | "reason": string                                                        |
| reject          | "reason": string                                                        |
| escalate        | "reason": string                                                        |

Issue types: amount_mismatch, missing_po, duplicate_invoice, vendor_mismatch,
date_anomaly, tax_error, missing_approval, over_budget, suspicious_vendor

## EXACT EXECUTION ORDER (follow this EVERY episode)

1. **categorize** — Use the "recommended_category" value from computed_analysis directly.

2. **set_priority** — Use the "recommended_priority" value from computed_analysis directly.

3. **extract_field** — extract ONLY the fields listed in "fields_to_extract" from computed_analysis.
   For each key→value pair in fields_to_extract, emit one extract_field action with field_name=key, field_value=value.
   Format rules: always use 2 decimal places for numbers. Copy the exact value from fields_to_extract.
   Do NOT extract any fields NOT listed in fields_to_extract — each unnecessary action costs -0.05.

4. **validate_match** — compare invoice vs PO:
   - If NO PO exists: skip this step entirely.
   - Read the "po_match_expected" boolean from computed_analysis. Use THAT value directly.

5. **flag_issue** — Read the "issues_to_flag" list from computed_analysis. Flag EACH issue in that list, and NO others.

6. **Final decision** — Read "recommended_decision" from computed_analysis. Use it directly.
   Use approve, reject, or escalate action type (NOT submit_decision).

## PRE-COMPUTED ANALYSIS
The observation includes a "computed_analysis" field with pre-calculated metrics.
Use these DIRECTLY — they are authoritative truth. Do NOT second-guess them.

## EFFICIENCY
Complete all actions in minimum steps. Each step costs -0.05.
""")


# ---------------------------------------------------------------------------
# Pre-computed analysis engine
# ---------------------------------------------------------------------------


def _compute_analysis(obs: InvoiceObservation) -> Dict[str, Any]:
    """Pre-compute metrics and authoritative issue/decision flags."""
    analysis: Dict[str, Any] = {}
    issues_to_flag: List[str] = []

    inv = obs.invoice
    po = obs.purchase_order

    if inv is None:
        return analysis

    analysis["invoice_subtotal"] = inv.subtotal
    analysis["invoice_total"] = inv.total_amount
    analysis["invoice_tax"] = inv.tax_amount

    # ---- Category detection ----
    all_desc = " ".join(li.description.lower() for li in inv.line_items)
    if inv.notes:
        all_desc += " " + inv.notes.lower()

    def _has_word(text: str, *keywords: str) -> bool:
        return any(re.search(r'\b' + re.escape(kw) + r'', text) for kw in keywords)

    category = "other"
    if _has_word(all_desc, "market", "seo", "ppc", "campaign", "print", "poster", "social media", "ad spend"):
        category = "marketing"
    elif _has_word(all_desc, "office", "paper", "pen", "supplies", "toner"):
        category = "supplies"
    elif _has_word(all_desc, "consult", "architect", "review", "audit", "advisory"):
        category = "consulting"
    elif _has_word(all_desc, "maint", "grounds", "repair", "clean", "tree", "landscap", "mulch", "removal"):
        category = "maintenance"
    elif _has_word(all_desc, "license", "software", "saas", "platform", "analytics"):
        category = "software"
    elif _has_word(all_desc, "travel", "flight", "hotel"):
        category = "travel"
    elif _has_word(all_desc, "electric", "gas", "water", "utility"):
        category = "utilities"
    elif _has_word(all_desc, "equipment", "hardware", "server"):
        category = "equipment"
    analysis["recommended_category"] = category

    # ---- Tax rate analysis ----
    tax_error = False
    if inv.subtotal > 0:
        expected_tax = inv.subtotal * 0.08
        tax_error = abs(inv.tax_amount - expected_tax) > (inv.subtotal * 0.005)
        analysis["tax_error_detected"] = tax_error

    # ---- Date analysis ----
    date_anomaly = False
    if inv.invoice_date and inv.due_date:
        try:
            inv_date = datetime.strptime(inv.invoice_date, "%Y-%m-%d")
            due_date = datetime.strptime(inv.due_date, "%Y-%m-%d")
            days_diff = (due_date - inv_date).days
            date_anomaly = days_diff < 7
            analysis["date_anomaly_detected"] = date_anomaly
        except ValueError:
            pass

    # ---- PO comparison ----
    amount_mismatch = False
    over_budget = False
    vendor_mismatch = False

    if po is not None:
        diff = inv.subtotal - po.total_amount
        has_price_increase = False
        for i, inv_line in enumerate(inv.line_items):
            if i < len(po.items):
                price_diff = inv_line.unit_price - po.items[i].unit_price
                if price_diff > 0.01:
                    has_price_increase = True
            else:
                has_price_increase = True

        amount_mismatch = diff > 1.0 or has_price_increase
        analysis["amount_mismatch_detected"] = amount_mismatch

        over_budget = inv.total_amount > po.remaining_budget
        analysis["over_budget_detected"] = over_budget

        vendor_mismatch = inv.vendor_name != po.vendor_name
        analysis["vendor_mismatch_detected"] = vendor_mismatch
    else:
        analysis["no_purchase_order"] = True
        analysis["missing_po_detected"] = inv.po_number is None

    # ---- Duplicate / historical analysis ----
    duplicate_detected = False
    suspicious_vendor = False

    if obs.historical_invoices:
        duplicates = []
        same_po_overlaps = []
        subtotals_over_time = []

        for hist in obs.historical_invoices:
            if (
                abs(hist.total_amount - inv.total_amount) < 0.01
                and hist.vendor_id == inv.vendor_id
                and hist.invoice_id != inv.invoice_id
            ):
                duplicates.append(hist.invoice_id)
            if (
                hist.po_number and inv.po_number
                and hist.po_number == inv.po_number
                and hist.invoice_id != inv.invoice_id
            ):
                same_po_overlaps.append(hist.invoice_id)
            subtotals_over_time.append({
                "date": hist.invoice_date,
                "subtotal": hist.subtotal,
            })

        if duplicates:
            duplicate_detected = True
        elif same_po_overlaps:
            duplicate_detected = True
        analysis["duplicate_detected"] = duplicate_detected

        subtotals_over_time.sort(key=lambda x: x["date"])
        subtotals_over_time.append({"date": inv.invoice_date, "subtotal": inv.subtotal})

        if len(subtotals_over_time) >= 2:
            prev = subtotals_over_time[0]["subtotal"]
            escalations = 0
            for entry in subtotals_over_time[1:]:
                curr = entry["subtotal"]
                if prev > 0 and ((curr - prev) / prev) * 100 >= 30:
                    escalations += 1
                prev = curr
            if escalations >= 2:
                suspicious_vendor = True
        analysis["suspicious_vendor_detected"] = suspicious_vendor

    # Build issues list
    if amount_mismatch:
        issues_to_flag.append("amount_mismatch")
    if po is None and inv.po_number is None:
        issues_to_flag.append("missing_po")
    if duplicate_detected:
        issues_to_flag.append("duplicate_invoice")
    if vendor_mismatch:
        issues_to_flag.append("vendor_mismatch")
    if date_anomaly:
        issues_to_flag.append("date_anomaly")
    if tax_error:
        issues_to_flag.append("tax_error")
    if over_budget:
        issues_to_flag.append("over_budget")
    if suspicious_vendor:
        issues_to_flag.append("suspicious_vendor")
    analysis["issues_to_flag"] = issues_to_flag

    # ---- Priority ----
    if inv.total_amount > 30000 or len(issues_to_flag) >= 3:
        priority = "urgent"
    elif inv.total_amount > 5000 or amount_mismatch or duplicate_detected:
        priority = "high"
    elif inv.total_amount > 1000 or (po is None and inv.po_number is None):
        priority = "medium"
    else:
        priority = "low"
    analysis["recommended_priority"] = priority

    # ---- PO match ----
    po_match_expected = True
    if po is not None:
        has_any_issues = len(issues_to_flag) > 0
        po_match_expected = (not has_any_issues) or ("amount_mismatch" not in issues_to_flag)
    analysis["po_match_expected"] = po_match_expected

    # ---- Decision ----
    if po is None and inv.po_number is None:
        recommended = "reject"
    elif duplicate_detected and not amount_mismatch and not suspicious_vendor:
        recommended = "reject"
    elif len(issues_to_flag) >= 3:
        hard_fraud = vendor_mismatch or date_anomaly or tax_error or duplicate_detected
        if hard_fraud:
            recommended = "reject"
        else:
            recommended = "escalate"
    elif amount_mismatch and not duplicate_detected:
        recommended = "escalate"
    elif len(issues_to_flag) == 0:
        recommended = "approve"
    else:
        recommended = "escalate"
    analysis["recommended_decision"] = recommended

    # ---- Fields to extract ----
    expected_extractions = {
        "vendor_name": inv.vendor_name,
        "total_amount": f"{inv.total_amount:.2f}",
    }
    if inv.po_number:
        expected_extractions["po_number"] = inv.po_number
    if tax_error:
        expected_extractions["tax_amount"] = f"{inv.tax_amount:.2f}"
    analysis["fields_to_extract"] = expected_extractions

    return analysis


# ---------------------------------------------------------------------------
# Observation → prompt context
# ---------------------------------------------------------------------------


def obs_to_context(obs: InvoiceObservation, step_num: int) -> str:
    """Serialize observation with pre-computed analysis."""
    data: Dict[str, Any] = {
        "step": f"{obs.step_number}/{obs.max_steps}",
        "progress": obs.progress,
        "available_actions": obs.available_actions,
    }
    if step_num <= 1:
        data["goal"] = obs.goal
        if obs.invoice:
            data["invoice"] = obs.invoice.model_dump()
        if obs.purchase_order:
            data["purchase_order"] = obs.purchase_order.model_dump()
        if obs.historical_invoices:
            data["historical_invoices"] = [h.model_dump() for h in obs.historical_invoices]
        data["computed_analysis"] = _compute_analysis(obs)
    if obs.last_action_feedback:
        data["last_action_feedback"] = obs.last_action_feedback
    if obs.last_action_error:
        data["last_action_error"] = obs.last_action_error
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------


def parse_model_action(raw: str) -> InvoiceAction:
    """Parse the LLM JSON response into an InvoiceAction."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        json_lines = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            elif line.strip() == "```" and in_fence:
                break
            elif in_fence:
                json_lines.append(line)
        if json_lines:
            cleaned = "\n".join(json_lines)

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Handle thinking blocks
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    # Find JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    payload = json.loads(cleaned)
    if "action_type" not in payload:
        raise ValueError(f"Missing 'action_type': {payload}")

    action_type_str = payload["action_type"].lower().strip()
    action_type = _ACTION_TYPE_MAP.get(action_type_str)
    if action_type is None:
        raise ValueError(f"Unknown action_type '{action_type_str}'")

    kwargs: Dict[str, Any] = {"action_type": action_type}

    if "category" in payload and payload["category"] is not None:
        cat = _CATEGORY_MAP.get(payload["category"].lower().strip())
        if cat:
            kwargs["category"] = cat

    if "priority" in payload and payload["priority"] is not None:
        pri = _PRIORITY_MAP.get(payload["priority"].lower().strip())
        if pri:
            kwargs["priority"] = pri

    if "issue_type" in payload and payload["issue_type"] is not None:
        iss = _ISSUE_MAP.get(payload["issue_type"].lower().strip())
        if iss:
            kwargs["issue_type"] = iss

    for field in ("issue_description", "field_name", "field_value", "reason"):
        if field in payload and payload[field] is not None:
            kwargs[field] = str(payload[field])

    if "match_result" in payload and payload["match_result"] is not None:
        val = payload["match_result"]
        kwargs["match_result"] = val if isinstance(val, bool) else str(val).lower() == "true"

    return InvoiceAction(**kwargs)


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def compute_max_reward(task_id: str) -> float:
    """Compute the theoretical maximum reward for a task."""
    task = ALL_TASKS[task_id]
    n_issues = len(task.expected_issues)
    n_extracts = len(task.expected_extractions)
    has_po = task.purchase_order is not None

    steps = 1 + 1 + n_extracts + (1 if has_po else 0) + n_issues + 1

    r = REWARD_CORRECT_CATEGORY + REWARD_CORRECT_PRIORITY + REWARD_CORRECT_DECISION
    r += 1.0  # subtask completion bonus
    r += n_extracts * REWARD_CORRECT_EXTRACTION
    if has_po:
        r += REWARD_PO_MATCH_CORRECT
    r += n_issues * REWARD_CORRECT_ISSUE_FLAG
    r += steps * REWARD_STEP_COST
    return r


def normalize_score(raw_reward: float, task_id: str) -> float:
    """Normalize raw reward to 0.0–1.0 range."""
    max_r = compute_max_reward(task_id)
    if max_r <= 0:
        return 0.0
    return max(0.0, min(1.0, raw_reward / max_r))


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run inference on all 6 tasks and report scores."""
    if not HF_TOKEN:
        print(
            "ERROR: No API key found. Set HF_TOKEN, OPENAI_API_KEY, or API key env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL: {API_BASE_URL}")
    print(f"MODEL_NAME:   {MODEL_NAME}")
    print(f"HF_TOKEN:     {'*' * min(8, len(HF_TOKEN))}...{HF_TOKEN[-4:] if len(HF_TOKEN) > 4 else '****'}")
    print()

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    task_ids = list(ALL_TASKS.keys())
    results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        env = InvoiceTriageEnvironment(task_id=task_id)
        obs = env.reset(seed=42)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        print(f"{'='*70}")
        print(f"  TASK: {task_id}")
        print(f"  GOAL: {obs.goal}")
        print(f"{'='*70}")

        step = 0
        while not obs.done and step < MAX_STEPS:
            step += 1
            context = obs_to_context(obs, step)
            messages.append({"role": "user", "content": context})

            response_text = ""
            for attempt in range(4):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    break
                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = 20 * (attempt + 1)
                        print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt+2}/4...")
                        time.sleep(wait)
                    else:
                        print(f"  [LLM ERROR] {exc}. Using fallback.")
                        response_text = FALLBACK_ACTION
                        break
            else:
                print(f"  [LLM ERROR] All retries failed. Using fallback.")
                response_text = FALLBACK_ACTION

            try:
                action = parse_model_action(response_text)
                messages.append({"role": "assistant", "content": response_text})
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"  [PARSE ERROR] {exc}")
                # Retry once
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Parse error. Respond with ONLY valid JSON. No markdown.",
                })
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    action = parse_model_action(response_text)
                    messages.append({"role": "assistant", "content": response_text})
                except Exception:
                    print(f"  [FATAL] Could not parse after retry. Using fallback.")
                    action = parse_model_action(FALLBACK_ACTION)

            obs = env.step(action)

            fb = obs.last_action_feedback or ""
            err = obs.last_action_error or ""
            label = action.action_type.value.upper()
            print(f"  [{label:<18}] {fb}{(' ERROR: ' + err) if err else ''}")

            if obs.done:
                break

        # Environment now returns normalized 0.0-1.0 scores directly
        final_score = obs.reward
        correct = final_score > 0

        print(f"  SCORE: {final_score:.4f}  |  STEPS: {step}")
        print(f"  PROGRESS: {json.dumps(obs.progress, indent=2)}")

        results.append({
            "task_id": task_id,
            "score": final_score,
            "steps": step,
            "correct": correct,
        })

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (model={MODEL_NAME})")
    print(f"{'='*70}")
    for r in results:
        icon = "✓" if r["correct"] else "✗"
        print(
            f"  {icon} {r['task_id']:<35} "
            f"score={r['score']:.4f}  "
            f"steps={r['steps']}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    correct_count = sum(1 for r in results if r["correct"])

    print(f"\n  Avg score: {avg_score:.4f}")
    print(f"  Tasks correct: {correct_count}/{len(results)}")


if __name__ == "__main__":
    main()
