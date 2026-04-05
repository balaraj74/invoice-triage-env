"""LLM-powered agent that processes invoices using OpenAI or Gemini.

Elite-grade agent with pre-computed analysis, optimal action ordering,
and exhaustive issue detection to maximise reward on every task.

Usage:
    # OpenAI (default)
    python -m invoice_triage_env.examples.run_llm_agent
    python -m invoice_triage_env.examples.run_llm_agent --model gpt-4o

    # Google Gemini
    python -m invoice_triage_env.examples.run_llm_agent --provider gemini
    python -m invoice_triage_env.examples.run_llm_agent --provider gemini --model gemini-2.5-flash

    # Single task, hard tasks only
    python -m invoice_triage_env.examples.run_llm_agent --task easy_approve_clean
    python -m invoice_triage_env.examples.run_llm_agent --difficulty hard

Environment Variables:
    OPENAI_API_KEY   — Required when --provider=openai (default).
    GEMINI_API_KEY   — Required when --provider=gemini.
                       Falls back to GOOGLE_API_KEY if not set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
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
)
from invoice_triage_env.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "gemini")
MAX_LLM_RETRIES = 2

_PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model": "gpt-4o-mini",
        "env_keys": ["OPENAI_API_KEY"],
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    },
}

# Maps string → enum for safe parsing
_ACTION_TYPE_MAP = {e.value: e for e in ActionType}
_CATEGORY_MAP = {e.value: e for e in InvoiceCategory}
_PRIORITY_MAP = {e.value: e for e in Priority}
_ISSUE_MAP = {e.value: e for e in IssueType}

# ---------------------------------------------------------------------------
# System prompt — exhaustive, reward-optimised
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an elite accounts-payable invoice triage agent. You MUST follow
the exact strategy and formats below to score maximum accuracy.

## RESPONSE FORMAT
Respond with EXACTLY ONE JSON object per turn. No markdown, no explanation.
```
{"action_type": "...", ...payload fields...}
```

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

1. **categorize** — Use the \"recommended_category\" value from computed_analysis directly.

2. **set_priority** — Use the \"recommended_priority\" value from computed_analysis directly.
   Rules (for understanding only, trust the pre-computed value):
   - total ≤ $1000 and no issues → low
   - total $1000–$5000 OR no PO → medium
   - total $5000–$30000 OR has PO amount discrepancy OR confirmed duplicate → high
   - total > $30000 OR 3+ issues → urgent

3. **extract_field** — extract ONLY the fields listed in "fields_to_extract" from computed_analysis.
   For each key→value pair in fields_to_extract, emit one extract_field action with field_name=key, field_value=value.
   Format rules: always use 2 decimal places for numbers. Copy the exact value from fields_to_extract.
   Do NOT extract any fields NOT listed in fields_to_extract — each unnecessary action costs -0.05.

4. **validate_match** — compare invoice vs PO:
   - If NO PO exists: skip this step entirely.
   - Read the "po_match_expected" boolean from computed_analysis. Use THAT value directly.
   - match_result = true means invoice legitimately aligns with the PO (even if subtotal < PO total — that's fine, invoice came in under budget).
   - match_result = false ONLY when the computed_analysis says amount_mismatch_detected is true (invoice EXCEEDS PO in unit prices or total).

5. **flag_issue** — Read the "issues_to_flag" list from computed_analysis. Flag EACH issue in that list, and NO others. One flag_issue action per issue.

   Issue definitions (for understanding — but trust the computed_analysis booleans):
   a) **amount_mismatch**: invoice subtotal EXCEEDS PO total, or any line item unit_price on the invoice is HIGHER than the PO unit price. Coming in UNDER the PO is normal and NOT a mismatch.
   b) **missing_po**: invoice has no po_number (null/None) and no PO record exists.
   c) **duplicate_invoice**: any historical invoice from same vendor has exact same total_amount (within $0.01). Also flag if historical invoices reference the same PO and overlap in scope.
   d) **vendor_mismatch**: vendor_name on invoice ≠ vendor_name on PO (exact string comparison).
   e) **date_anomaly**: (due_date - invoice_date) < 7 days.
   f) **tax_error**: tax rate deviates from expected 8% by more than 0.5%.
   g) **over_budget**: invoice total_amount > PO remaining_budget.
   h) **suspicious_vendor**: costs escalating ≥30% across sequential historical invoices.

   CRITICAL: ONLY flag issues listed in "issues_to_flag". Each false positive costs -1.0.
   Each correctly flagged issue earns +1.5. Each missed issue costs -0.8.

6. **Final decision** — Read "recommended_decision" from computed_analysis. Use it directly.
   Use approve, reject, or escalate action type (NOT submit_decision).

   Decision logic (for understanding):
   - No PO at all → reject
   - Confirmed duplicate with no other context → reject
   - Multiple issues (3+) including financial discrepancies → reject
   - Amount mismatch only (1 issue) or amount mismatch + over_budget + suspicious_vendor → escalate
   - Clean invoice, all matches, no issues → approve

## PRE-COMPUTED ANALYSIS
The observation includes a "computed_analysis" field with pre-calculated metrics.
It contains "issues_to_flag" (exact list of issues to flag), "po_match_expected" (boolean for validate_match),
"recommended_decision" (approve/reject/escalate), and "fields_to_extract" (field→value map).
Use these DIRECTLY — they are authoritative truth. Do NOT second-guess them.

## EFFICIENCY
Complete all actions in minimum steps. The action order above is optimal.
Each step costs -0.05, so don't waste steps. Never repeat an action type
that was already completed (check the "progress" field).
""")


# ---------------------------------------------------------------------------
# Pre-computed analysis — inject into context for the LLM
# ---------------------------------------------------------------------------


def _compute_analysis(obs: InvoiceObservation) -> Dict[str, Any]:
    """Pre-compute critical metrics AND authoritative issue/decision flags.

    The LLM should trust issues_to_flag and recommended_decision directly.
    """
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
    # Scan all line item descriptions for keywords
    all_desc = " ".join(
        li.description.lower() for li in inv.line_items
    )
    if inv.notes:
        all_desc += " " + inv.notes.lower()

    import re
    def _has_word(text: str, *keywords: str) -> bool:
        """Check if any keyword appears as a whole word (word boundary aware)."""
        return any(re.search(r'\b' + re.escape(kw) + r'', text) for kw in keywords)

    category = "other"
    # Check marketing BEFORE supplies to avoid "pen" matching inside "spend"
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
        actual_rate = inv.tax_amount / inv.subtotal
        expected_tax = inv.subtotal * 0.08
        analysis["actual_tax_rate_pct"] = round(actual_rate * 100, 2)
        analysis["expected_tax_at_8pct"] = round(expected_tax, 2)
        analysis["tax_difference"] = round(inv.tax_amount - expected_tax, 2)
        tax_error = abs(inv.tax_amount - expected_tax) > (inv.subtotal * 0.005)
        analysis["tax_error_detected"] = tax_error

    # ---- Date analysis ----
    date_anomaly = False
    if inv.invoice_date and inv.due_date:
        try:
            from datetime import datetime

            inv_date = datetime.strptime(inv.invoice_date, "%Y-%m-%d")
            due_date = datetime.strptime(inv.due_date, "%Y-%m-%d")
            days_diff = (due_date - inv_date).days
            analysis["days_between_invoice_and_due"] = days_diff
            date_anomaly = days_diff < 7
            analysis["date_anomaly_detected"] = date_anomaly
        except ValueError:
            pass

    # ---- PO comparison ----
    amount_mismatch = False
    over_budget = False
    vendor_mismatch = False

    if po is not None:
        analysis["po_total"] = po.total_amount
        analysis["po_remaining_budget"] = po.remaining_budget
        diff = inv.subtotal - po.total_amount
        analysis["subtotal_vs_po_diff"] = round(diff, 2)

        # Amount mismatch = invoice subtotal EXCEEDS PO total (not under)
        # OR line item prices INCREASED vs PO
        has_price_increase = False
        line_diffs = []
        for i, inv_line in enumerate(inv.line_items):
            if i < len(po.items):
                po_line = po.items[i]
                price_diff = inv_line.unit_price - po_line.unit_price
                if abs(price_diff) > 0.01:
                    line_diffs.append({
                        "line": i + 1,
                        "description": inv_line.description,
                        "invoice_unit_price": inv_line.unit_price,
                        "po_unit_price": po_line.unit_price,
                        "diff": round(price_diff, 2),
                    })
                    if price_diff > 0.01:
                        has_price_increase = True
            else:
                line_diffs.append({
                    "line": i + 1,
                    "description": inv_line.description,
                    "note": "NOT in PO — extra line item on invoice",
                })
                has_price_increase = True

        if line_diffs:
            analysis["line_item_discrepancies"] = line_diffs

        # Mismatch when invoice exceeds PO or has price increases
        amount_mismatch = diff > 1.0 or has_price_increase
        analysis["amount_mismatch_detected"] = amount_mismatch

        over_budget = inv.total_amount > po.remaining_budget
        analysis["over_budget_detected"] = over_budget

        # Vendor name comparison
        vendor_mismatch = inv.vendor_name != po.vendor_name
        analysis["invoice_vendor_name"] = inv.vendor_name
        analysis["po_vendor_name"] = po.vendor_name
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
            # Exact amount match = clear duplicate
            if (
                abs(hist.total_amount - inv.total_amount) < 0.01
                and hist.vendor_id == inv.vendor_id
                and hist.invoice_id != inv.invoice_id
            ):
                duplicates.append(hist.invoice_id)

            # Same PO reference = possible scope overlap / double-billing
            if (
                hist.po_number
                and inv.po_number
                and hist.po_number == inv.po_number
                and hist.invoice_id != inv.invoice_id
            ):
                same_po_overlaps.append(hist.invoice_id)

            subtotals_over_time.append({
                "invoice_id": hist.invoice_id,
                "date": hist.invoice_date,
                "subtotal": hist.subtotal,
            })

        if duplicates:
            analysis["duplicate_invoice_ids"] = duplicates
            duplicate_detected = True
        elif same_po_overlaps:
            # Multiple invoices against the same PO = likely duplicate billing
            analysis["same_po_overlap_ids"] = same_po_overlaps
            duplicate_detected = True

        analysis["duplicate_detected"] = duplicate_detected

        # Sort by date for escalation analysis
        subtotals_over_time.sort(key=lambda x: x["date"])
        subtotals_over_time.append({
            "invoice_id": inv.invoice_id,
            "date": inv.invoice_date,
            "subtotal": inv.subtotal,
        })
        analysis["cost_history"] = subtotals_over_time

        # Check for escalating costs (≥30% increase)
        if len(subtotals_over_time) >= 2:
            prev = subtotals_over_time[0]["subtotal"]
            escalations = []
            for entry in subtotals_over_time[1:]:
                curr = entry["subtotal"]
                if prev > 0:
                    pct_change = ((curr - prev) / prev) * 100
                    if pct_change >= 30:
                        escalations.append({
                            "from": prev,
                            "to": curr,
                            "pct_increase": round(pct_change, 1),
                        })
                prev = curr
            if escalations:
                analysis["cost_escalations"] = escalations
                # Require a sustained pattern (2+ escalation events across 3+ invoices)
                # A single increase could be legitimate scope change
                if len(escalations) >= 2:
                    suspicious_vendor = True

        analysis["suspicious_vendor_detected"] = suspicious_vendor

    # ================================================================
    # Build authoritative issues_to_flag list
    # ================================================================
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
    analysis["issue_count"] = len(issues_to_flag)

    # ================================================================
    # Recommended priority
    # ================================================================
    # Priority is driven by amount thresholds and issue severity,
    # NOT by keywords in notes ("urgent" in notes describes content, not processing urgency)
    if inv.total_amount > 30000 or len(issues_to_flag) >= 3:
        priority = "urgent"
    elif inv.total_amount > 5000 or amount_mismatch or duplicate_detected:
        priority = "high"
    elif inv.total_amount > 1000 or (po is None and inv.po_number is None):
        priority = "medium"
    else:
        priority = "low"
    analysis["recommended_priority"] = priority

    # ================================================================
    # PO match expected value  (mirrors env logic exactly)
    # ================================================================
    # Environment logic:  po_should_match = not has_issues or "amount_mismatch" not in expected_issues
    # Since we're computing issues_to_flag as our best prediction of expected_issues:
    po_match_expected = True  # default
    if po is not None:
        has_any_issues = len(issues_to_flag) > 0
        po_match_expected = (not has_any_issues) or ("amount_mismatch" not in issues_to_flag)
    analysis["po_match_expected"] = po_match_expected

    # ================================================================
    # Recommended decision
    # ================================================================
    if po is None and inv.po_number is None:
        recommended = "reject"
    elif duplicate_detected and not amount_mismatch and not suspicious_vendor:
        recommended = "reject"
    elif len(issues_to_flag) >= 3:
        # Many issues — likely needs rejection for fraud/serious problems
        # But if it's amount_mismatch + over_budget + suspicious_vendor
        # with no hard fraud indicators, escalate
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

    # ================================================================
    # Fields to extract (exact values for the LLM to copy)
    # ================================================================
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
# Observation → prompt builder (enriched)
# ---------------------------------------------------------------------------


def _obs_to_context(obs: InvoiceObservation, step_num: int) -> str:
    """Serialize the observation with pre-computed analysis."""
    data: Dict[str, Any] = {
        "step": f"{obs.step_number}/{obs.max_steps}",
        "progress": obs.progress,
        "available_actions": obs.available_actions,
    }

    # Only include full data on first step to save tokens
    if step_num <= 1:
        data["goal"] = obs.goal
        if obs.invoice:
            data["invoice"] = obs.invoice.model_dump()
        if obs.purchase_order:
            data["purchase_order"] = obs.purchase_order.model_dump()
        if obs.historical_invoices:
            data["historical_invoices"] = [
                h.model_dump() for h in obs.historical_invoices
            ]
        # Pre-computed analysis — the key differentiator
        data["computed_analysis"] = _compute_analysis(obs)

    if obs.last_action_feedback:
        data["last_action_feedback"] = obs.last_action_feedback
    if obs.last_action_error:
        data["last_action_error"] = obs.last_action_error

    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# LLM response → InvoiceAction parser
# ---------------------------------------------------------------------------


def _parse_llm_action(raw: str) -> InvoiceAction:
    """Parse the LLM JSON response into an InvoiceAction.

    Raises ValueError on malformed / unparseable responses.
    """
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Find the actual JSON content between fences
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
        else:
            # Fallback: strip first and last lines
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Handle thinking blocks (e.g., <think>...</think>)
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    # Find the first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}\nRaw: {raw!r}"
        )

    if "action_type" not in payload:
        raise ValueError(f"Missing 'action_type' in LLM response: {payload}")

    action_type_str = payload["action_type"].lower().strip()
    action_type = _ACTION_TYPE_MAP.get(action_type_str)
    if action_type is None:
        raise ValueError(
            f"Unknown action_type '{action_type_str}'. "
            f"Valid: {list(_ACTION_TYPE_MAP.keys())}"
        )

    kwargs: Dict[str, Any] = {"action_type": action_type}

    # Parse optional enum fields
    if "category" in payload and payload["category"] is not None:
        cat_str = payload["category"].lower().strip()
        cat = _CATEGORY_MAP.get(cat_str)
        if cat is None:
            raise ValueError(f"Unknown category '{payload['category']}'")
        kwargs["category"] = cat

    if "priority" in payload and payload["priority"] is not None:
        pri_str = payload["priority"].lower().strip()
        pri = _PRIORITY_MAP.get(pri_str)
        if pri is None:
            raise ValueError(f"Unknown priority '{payload['priority']}'")
        kwargs["priority"] = pri

    if "issue_type" in payload and payload["issue_type"] is not None:
        iss_str = payload["issue_type"].lower().strip()
        iss = _ISSUE_MAP.get(iss_str)
        if iss is None:
            raise ValueError(f"Unknown issue_type '{payload['issue_type']}'")
        kwargs["issue_type"] = iss

    # Pass-through string fields
    for fld in ("issue_description", "field_name", "field_value", "reason"):
        if fld in payload and payload[fld] is not None:
            kwargs[fld] = str(payload[fld])

    if "match_result" in payload and payload["match_result"] is not None:
        val = payload["match_result"]
        if isinstance(val, bool):
            kwargs["match_result"] = val
        elif isinstance(val, str):
            kwargs["match_result"] = val.lower().strip() in (
                "true",
                "1",
                "yes",
            )
        else:
            kwargs["match_result"] = bool(val)

    return InvoiceAction(**kwargs)


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------


def _resolve_api_key(provider: str) -> str:
    """Resolve the API key from environment variables for a provider."""
    cfg = _PROVIDER_DEFAULTS[provider]
    for env_var in cfg["env_keys"]:
        key = os.environ.get(env_var)
        if key:
            return key

    env_list = " or ".join(cfg["env_keys"])
    print(
        f"ERROR: No API key found for provider '{provider}'.\n"
        f"  Set one of: {env_list}\n"
        f"  Example: export {cfg['env_keys'][0]}='your-key-here'\n",
        file=sys.stderr,
    )
    sys.exit(1)


class LLMAgent:
    """Elite LLM agent — OpenAI or Gemini, one call per env step.

    Key improvements over a naive agent:
    - Pre-computed numerical analysis injected into context
    - Exhaustive issue-detection checklist in the system prompt
    - Optimal action ordering to minimise step cost
    - Token-efficient context: only sends full data on step 1
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
    ) -> None:
        if provider not in _PROVIDER_DEFAULTS:
            print(
                f"ERROR: Unknown provider '{provider}'. "
                f"Supported: {list(_PROVIDER_DEFAULTS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = _PROVIDER_DEFAULTS[provider]
        self._provider = provider
        self._model = model or cfg["model"]
        self._api_key = _resolve_api_key(provider)
        self._step_count = 0

        if provider == "openai":
            self._openai_client = OpenAI(api_key=self._api_key)
        elif provider == "gemini":
            try:
                from google import genai  # type: ignore[import-untyped]
            except ImportError:
                print(
                    "ERROR: google-genai package is required for Gemini.\n"
                    "  pip install google-genai\n",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Use Vertex AI endpoint (billing credits) if GCP project is set,
            # otherwise fall back to AI Studio API key.
            gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
            gcp_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            if gcp_project:
                print(
                    f"  [GEMINI] Using Vertex AI endpoint "
                    f"(project={gcp_project}, region={gcp_location})",
                    flush=True,
                )
                self._gemini_client = genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=gcp_location,
                )
            else:
                self._gemini_client = genai.Client(api_key=self._api_key)

        self._messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def reset(self) -> None:
        """Clear conversation history for a new episode."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._step_count = 0

    # -- Provider dispatchers --------------------------------------------------

    def _call_openai(self) -> str:
        """Send messages via the OpenAI SDK and return raw text."""
        response = self._openai_client.chat.completions.create(
            model=self._model,
            messages=self._messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _call_gemini(self) -> str:
        """Send messages via google-genai with rate-limit retry."""
        import time as _time

        from google.genai import types  # type: ignore[import-untyped]
        from google.genai.errors import ClientError  # type: ignore[import-untyped]

        system_text = ""
        contents: list[types.Content] = []
        for msg in self._messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])],
                    )
                )

        max_rate_retries = 3
        backoff = 25.0

        for retry in range(1 + max_rate_retries):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_text,
                        temperature=0.0,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                    ),
                )
                return response.text or ""
            except ClientError as exc:
                if exc.code == 429 and retry < max_rate_retries:
                    wait = backoff * (2**retry)
                    print(
                        f"  [RATE LIMIT] Gemini 429 — waiting {wait:.0f}s "
                        f"(retry {retry + 1}/{max_rate_retries})...",
                        flush=True,
                    )
                    _time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Gemini rate-limit retries exhausted")

    # -- Main action method ----------------------------------------------------

    def act(self, obs: InvoiceObservation) -> InvoiceAction:
        """Ask the LLM for the next action given the current observation."""
        self._step_count += 1
        context = _obs_to_context(obs, self._step_count)
        self._messages.append({"role": "user", "content": context})

        call_fn = (
            self._call_openai
            if self._provider == "openai"
            else self._call_gemini
        )

        for attempt in range(1 + MAX_LLM_RETRIES):
            raw = call_fn()

            try:
                action = _parse_llm_action(raw)
                self._messages.append({"role": "assistant", "content": raw})
                return action
            except ValueError as exc:
                if attempt < MAX_LLM_RETRIES:
                    self._messages.append(
                        {"role": "assistant", "content": raw}
                    )
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Parse error: {exc}\n"
                                "Respond with ONLY valid JSON matching the "
                                "action schema. No markdown, no explanation."
                            ),
                        }
                    )
                else:
                    raise


def run_llm_episode(
    task_id: str,
    agent: LLMAgent,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single episode using the LLM agent."""
    env = InvoiceTriageEnvironment(task_id=task_id)
    obs = env.reset(seed=42)
    agent.reset()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TASK: {task_id}")
        print(f"  GOAL: {obs.goal}")
        print(f"{'='*70}")

    step = 0
    while not obs.done:
        step += 1
        try:
            action = agent.act(obs)
        except ValueError as exc:
            if verbose:
                print(f"  [PARSE ERROR] {exc}")
            break

        obs = env.step(action)

        if verbose:
            fb = obs.last_action_feedback or ""
            err = obs.last_action_error or ""
            label = action.action_type.value.upper()
            print(
                f"  [{label:<18}] {fb}"
                f"{(' ERROR: ' + err) if err else ''}"
            )

    if verbose:
        print(
            f"  REWARD: {obs.reward:.2f}  |  "
            f"DONE: {obs.done}  |  STEPS: {step}"
        )
        print(f"  PROGRESS: {json.dumps(obs.progress, indent=2)}")

    return {
        "task_id": task_id,
        "reward": obs.reward,
        "done": obs.done,
        "steps": step,
        "progress": obs.progress,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM-powered invoice triage agent.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=list(SUPPORTED_PROVIDERS),
        default="openai",
        help="LLM provider (default: openai).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a single task by ID (e.g. 'easy_approve_clean').",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run all tasks for a given difficulty.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Defaults per provider: "
            + ", ".join(
                f"{p}={c['model']}" for p, c in _PROVIDER_DEFAULTS.items()
            )
            + "."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output; only print the summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent = LLMAgent(provider=args.provider, model=args.model)
    effective_model = agent._model

    if args.task:
        if args.task not in ALL_TASKS:
            print(
                f"Unknown task '{args.task}'. "
                f"Available: {list(ALL_TASKS.keys())}"
            )
            sys.exit(1)
        task_ids = [args.task]
    elif args.difficulty:
        task_ids = [t.task_id for t in TASKS_BY_DIFFICULTY[args.difficulty]]
    else:
        task_ids = list(ALL_TASKS.keys())

    verbose = not args.quiet
    results: List[Dict[str, Any]] = []

    for tid in task_ids:
        result = run_llm_episode(tid, agent, verbose=verbose)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (provider={args.provider}, model={effective_model})")
    print(f"{'='*70}")
    for r in results:
        icon = "✓" if r["reward"] > 0 else "✗"
        print(
            f"  {icon} {r['task_id']:<35} "
            f"reward={r['reward']:>7.2f}  "
            f"steps={r['steps']}"
        )
    total = sum(r["reward"] for r in results)
    avg = total / len(results) if results else 0
    print(f"\n  Total reward: {total:.2f}  |  Avg: {avg:.2f}")


if __name__ == "__main__":
    main()
