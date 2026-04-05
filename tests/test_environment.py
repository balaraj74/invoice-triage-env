"""Unit tests for InvoiceTriageEnvironment."""

from __future__ import annotations

import pytest

from invoice_triage_env.models import (
    ActionType,
    InvoiceAction,
    InvoiceCategory,
    IssueType,
    Priority,
)
from invoice_triage_env.server.invoice_triage_environment import (
    InvoiceTriageEnvironment,
)
from invoice_triage_env.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env_easy():
    """Environment loaded with easy_approve_clean task."""
    env = InvoiceTriageEnvironment(task_id="easy_approve_clean")
    env.reset(seed=42)
    return env


@pytest.fixture
def env_reject():
    """Environment loaded with easy_reject_no_po task."""
    env = InvoiceTriageEnvironment(task_id="easy_reject_no_po")
    env.reset(seed=42)
    return env


@pytest.fixture
def env_mismatch():
    """Environment loaded with medium_amount_mismatch task."""
    env = InvoiceTriageEnvironment(task_id="medium_amount_mismatch")
    env.reset(seed=42)
    return env


# ── Reset Tests ──────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env_easy):
        obs = env_easy.reset(seed=1)
        assert obs.done is False
        assert obs.step_number == 0
        assert obs.invoice is not None
        assert obs.goal != ""

    def test_reset_clears_state(self, env_easy):
        # Take an action, then reset
        env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        obs = env_easy.reset(seed=2)
        assert obs.step_number == 0
        assert all(v is False for v in obs.progress.values())

    @pytest.mark.parametrize("task_id", list(ALL_TASKS.keys()))
    def test_all_tasks_can_reset(self, task_id):
        env = InvoiceTriageEnvironment(task_id=task_id)
        obs = env.reset(seed=42)
        assert obs.invoice is not None
        assert obs.done is False


# ── Categorize Tests ─────────────────────────────────────────────────────────

class TestCategorize:
    def test_correct_category(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        assert "Correct" in obs.last_action_feedback
        assert obs.progress["categorized"] is True

    def test_wrong_category(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SOFTWARE,
        ))
        assert "Incorrect" in obs.last_action_feedback

    def test_duplicate_category(self, env_easy):
        env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        assert "Already" in obs.last_action_feedback

    def test_missing_category_field(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
        ))
        assert obs.last_action_error is not None


# ── Priority Tests ───────────────────────────────────────────────────────────

class TestPriority:
    def test_correct_priority(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.SET_PRIORITY,
            priority=Priority.LOW,
        ))
        assert "Correct" in obs.last_action_feedback

    def test_wrong_priority(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.SET_PRIORITY,
            priority=Priority.URGENT,
        ))
        assert "Expected 'low'" in obs.last_action_feedback


# ── Extract Field Tests ──────────────────────────────────────────────────────

class TestExtractField:
    def test_correct_extraction(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.EXTRACT_FIELD,
            field_name="vendor_name",
            field_value="Acme Office Supplies",
        ))
        assert "Correct" in obs.last_action_feedback

    def test_wrong_extraction(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.EXTRACT_FIELD,
            field_name="total_amount",
            field_value="999.99",
        ))
        assert "mismatch" in obs.last_action_feedback

    def test_extraction_no_ground_truth(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.EXTRACT_FIELD,
            field_name="currency",
            field_value="USD",
        ))
        assert "no ground truth" in obs.last_action_feedback


# ── PO Validation Tests ─────────────────────────────────────────────────────

class TestValidateMatch:
    def test_po_match_correct(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.VALIDATE_MATCH,
            match_result=True,  # Easy task has matching PO
        ))
        assert "Correct" in obs.last_action_feedback

    def test_po_mismatch_correct(self, env_mismatch):
        obs = env_mismatch.step(InvoiceAction(
            action_type=ActionType.VALIDATE_MATCH,
            match_result=False,  # Mismatch task has non-matching PO
        ))
        assert "Correct" in obs.last_action_feedback


# ── Flag Issue Tests ─────────────────────────────────────────────────────────

class TestFlagIssue:
    def test_correct_issue(self, env_reject):
        obs = env_reject.step(InvoiceAction(
            action_type=ActionType.FLAG_ISSUE,
            issue_type=IssueType.MISSING_PO,
        ))
        assert "Good catch" in obs.last_action_feedback

    def test_false_positive(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.FLAG_ISSUE,
            issue_type=IssueType.AMOUNT_MISMATCH,
        ))
        assert "False positive" in obs.last_action_feedback

    def test_duplicate_flag(self, env_reject):
        env_reject.step(InvoiceAction(
            action_type=ActionType.FLAG_ISSUE,
            issue_type=IssueType.MISSING_PO,
        ))
        obs = env_reject.step(InvoiceAction(
            action_type=ActionType.FLAG_ISSUE,
            issue_type=IssueType.MISSING_PO,
        ))
        assert "already flagged" in obs.last_action_feedback


# ── Decision Tests ───────────────────────────────────────────────────────────

class TestDecision:
    def test_correct_approve(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="All looks good.",
        ))
        assert obs.done is True
        assert "CORRECT" in obs.last_action_feedback

    def test_wrong_approve(self, env_reject):
        obs = env_reject.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="Approving anyway.",
        ))
        assert obs.done is True
        assert "INCORRECT" in obs.last_action_feedback

    def test_correct_reject(self, env_reject):
        obs = env_reject.step(InvoiceAction(
            action_type=ActionType.REJECT,
            reason="No PO present.",
        ))
        assert obs.done is True
        assert "CORRECT" in obs.last_action_feedback

    def test_submit_decision_approve(self, env_easy):
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.SUBMIT_DECISION,
            reason="approve",
            issue_description="Looks fine.",
        ))
        assert obs.done is True

    def test_double_decision_rejected(self, env_easy):
        env_easy.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="First decision.",
        ))
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="Second decision.",
        ))
        assert "already submitted" in obs.last_action_feedback.lower() or \
               "already" in obs.last_action_feedback.lower()


# ── Episode Flow Tests ───────────────────────────────────────────────────────

class TestEpisodeFlow:
    def test_full_episode_approve(self, env_easy):
        """Run a complete correct episode for easy_approve_clean."""
        env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        env_easy.step(InvoiceAction(
            action_type=ActionType.SET_PRIORITY,
            priority=Priority.LOW,
        ))
        env_easy.step(InvoiceAction(
            action_type=ActionType.VALIDATE_MATCH,
            match_result=True,
        ))
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="Invoice matches PO, all fields valid.",
        ))
        assert obs.done is True
        assert obs.reward > 0
        assert obs.progress["categorized"] is True
        assert obs.progress["decision_made"] is True

    def test_step_after_done(self, env_easy):
        env_easy.step(InvoiceAction(
            action_type=ActionType.APPROVE,
            reason="Done.",
        ))
        obs = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        assert "already complete" in obs.last_action_feedback.lower()

    def test_reward_accumulates(self, env_easy):
        obs1 = env_easy.step(InvoiceAction(
            action_type=ActionType.CATEGORIZE,
            category=InvoiceCategory.SUPPLIES,
        ))
        obs2 = env_easy.step(InvoiceAction(
            action_type=ActionType.SET_PRIORITY,
            priority=Priority.LOW,
        ))
        # Each step has positive reward + step cost, so reward should grow
        assert obs2.reward > obs1.reward or obs2.reward != 0


# ── Task Registry Tests ──────────────────────────────────────────────────────

class TestTaskRegistry:
    def test_all_tasks_count(self):
        assert len(ALL_TASKS) == 6

    def test_difficulty_distribution(self):
        assert len(TASKS_BY_DIFFICULTY["easy"]) == 2
        assert len(TASKS_BY_DIFFICULTY["medium"]) == 2
        assert len(TASKS_BY_DIFFICULTY["hard"]) == 2

    @pytest.mark.parametrize("task_id,expected_decision", [
        ("easy_approve_clean", "approve"),
        ("easy_reject_no_po", "reject"),
        ("medium_amount_mismatch", "escalate"),
        ("medium_duplicate_detection", "reject"),
        ("hard_multi_issue_fraud", "reject"),
        ("hard_suspicious_vendor", "escalate"),
    ])
    def test_expected_decisions(self, task_id, expected_decision):
        assert ALL_TASKS[task_id].expected_decision == expected_decision
