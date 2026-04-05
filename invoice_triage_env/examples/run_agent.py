"""Example: rule-based agent that processes invoices locally (no server).

Demonstrates how to interact with InvoiceTriageEnvironment directly,
going through the full triage loop for every task.
"""

from __future__ import annotations

import json
from typing import Optional

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
from invoice_triage_env.tasks import ALL_TASKS


def _guess_category(vendor_name: str, notes: Optional[str]) -> InvoiceCategory:
    """Naive keyword-based categorizer."""
    text = f"{vendor_name} {notes or ''}".lower()
    if any(w in text for w in ("supply", "paper", "pen", "office")):
        return InvoiceCategory.SUPPLIES
    if any(w in text for w in ("consult", "architect", "advisory")):
        return InvoiceCategory.CONSULTING
    if any(w in text for w in ("market", "seo", "ppc", "campaign", "print")):
        return InvoiceCategory.MARKETING
    if any(w in text for w in ("maint", "grounds", "repair", "clean")):
        return InvoiceCategory.MAINTENANCE
    if any(w in text for w in ("license", "software", "saas", "platform")):
        return InvoiceCategory.SOFTWARE
    if any(w in text for w in ("electric", "gas", "water", "utility")):
        return InvoiceCategory.UTILITIES
    return InvoiceCategory.OTHER


def run_agent_episode(task_id: str) -> dict:
    """Run a single episode with the rule-based agent."""
    env = InvoiceTriageEnvironment(task_id=task_id)
    obs = env.reset(seed=42)

    print(f"\n{'='*70}")
    print(f"  TASK: {task_id}")
    print(f"  GOAL: {obs.goal}")
    print(f"{'='*70}")

    # Step 1: Categorize
    category = _guess_category(
        obs.invoice.vendor_name if obs.invoice else "",
        obs.invoice.notes if obs.invoice else None,
    )
    obs = env.step(InvoiceAction(
        action_type=ActionType.CATEGORIZE,
        category=category,
    ))
    print(f"  [CATEGORIZE] {obs.last_action_feedback}")

    # Step 2: Set priority
    priority = Priority.LOW
    if obs.invoice and obs.invoice.total_amount > 10000:
        priority = Priority.HIGH
    if obs.invoice and obs.invoice.total_amount > 30000:
        priority = Priority.URGENT
    obs = env.step(InvoiceAction(
        action_type=ActionType.SET_PRIORITY,
        priority=priority,
    ))
    print(f"  [PRIORITY]   {obs.last_action_feedback}")

    # Step 3: Extract key fields
    if obs.invoice:
        for field_name, field_value in [
            ("vendor_name", obs.invoice.vendor_name),
            ("total_amount", f"{obs.invoice.total_amount:.2f}"),
            ("po_number", obs.invoice.po_number or "N/A"),
        ]:
            obs = env.step(InvoiceAction(
                action_type=ActionType.EXTRACT_FIELD,
                field_name=field_name,
                field_value=field_value,
            ))
            print(f"  [EXTRACT]    {obs.last_action_feedback}")

    # Step 4: Validate PO match
    if obs.purchase_order and obs.invoice:
        amounts_match = abs(
            obs.invoice.subtotal - obs.purchase_order.total_amount
        ) < 1.0
        obs = env.step(InvoiceAction(
            action_type=ActionType.VALIDATE_MATCH,
            match_result=amounts_match,
        ))
        print(f"  [PO MATCH]   {obs.last_action_feedback}")

        # Step 5: Flag issues if amounts don't match
        if not amounts_match:
            obs = env.step(InvoiceAction(
                action_type=ActionType.FLAG_ISSUE,
                issue_type=IssueType.AMOUNT_MISMATCH,
                issue_description=(
                    f"Invoice subtotal ${obs.invoice.subtotal:.2f} "
                    f"exceeds PO total ${obs.purchase_order.total_amount:.2f}"
                ),
            ))
            print(f"  [FLAG]       {obs.last_action_feedback}")
    elif obs.invoice and obs.invoice.po_number is None:
        obs = env.step(InvoiceAction(
            action_type=ActionType.FLAG_ISSUE,
            issue_type=IssueType.MISSING_PO,
            issue_description="No purchase order linked to this invoice.",
        ))
        print(f"  [FLAG]       {obs.last_action_feedback}")

    # Step 6: Check for duplicates
    if obs.historical_invoices and obs.invoice:
        for hist in obs.historical_invoices:
            if (
                abs(hist.total_amount - obs.invoice.total_amount) < 0.01
                and hist.vendor_id == obs.invoice.vendor_id
                and hist.invoice_id != obs.invoice.invoice_id
            ):
                obs = env.step(InvoiceAction(
                    action_type=ActionType.FLAG_ISSUE,
                    issue_type=IssueType.DUPLICATE_INVOICE,
                    issue_description=(
                        f"Matches historical invoice {hist.invoice_id}"
                    ),
                ))
                print(f"  [FLAG]       {obs.last_action_feedback}")
                break

    # Step 7: Final decision
    issues = [k for k, v in obs.progress.items() if k == "issue_flagged" and v]
    has_issues = any(
        obs.last_action_feedback
        and "False positive" not in obs.last_action_feedback
        for _ in [1]
    )

    # Simple heuristic
    if obs.purchase_order is None and obs.invoice and obs.invoice.po_number is None:
        decision = "reject"
    elif obs.progress.get("issue_flagged", False):
        decision = "escalate"
    else:
        decision = "approve"

    obs = env.step(InvoiceAction(
        action_type=ActionType.SUBMIT_DECISION,
        reason=decision,
        issue_description=f"Agent decided to {decision} based on triage results.",
    ))
    print(f"  [DECISION]   {obs.last_action_feedback}")
    print(f"  REWARD: {obs.reward:.2f}  |  DONE: {obs.done}")
    print(f"  PROGRESS: {json.dumps(obs.progress, indent=2)}")

    return {
        "task_id": task_id,
        "reward": obs.reward,
        "done": obs.done,
        "steps": obs.step_number,
        "progress": obs.progress,
    }


def main() -> None:
    """Run the agent on all available tasks."""
    results = []
    for task_id in ALL_TASKS:
        result = run_agent_episode(task_id)
        results.append(result)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "✓" if r["reward"] > 0 else "✗"
        print(
            f"  {status} {r['task_id']:<35} "
            f"reward={r['reward']:>7.2f}  "
            f"steps={r['steps']}"
        )
    total = sum(r["reward"] for r in results)
    print(f"\n  Total reward: {total:.2f}")


if __name__ == "__main__":
    main()
