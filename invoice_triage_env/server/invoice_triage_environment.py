"""InvoiceTriageEnvironment — server-side environment logic.

Implements the OpenEnv Environment interface for invoice processing RL training.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from invoice_triage_env.models import (
    ActionType,
    InvoiceAction,
    InvoiceCategory,
    InvoiceObservation,
    InvoiceState,
    IssueType,
    Priority,
)
from invoice_triage_env.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY, TaskDefinition


# Reward constants
REWARD_CORRECT_CATEGORY = 1.0
REWARD_WRONG_CATEGORY = -0.5
REWARD_CORRECT_PRIORITY = 0.5
REWARD_WRONG_PRIORITY = -0.3
REWARD_CORRECT_ISSUE_FLAG = 1.5
REWARD_FALSE_POSITIVE_ISSUE = -1.0
REWARD_MISSED_ISSUE_PENALTY = -0.8  # applied at decision time
REWARD_CORRECT_DECISION = 3.0
REWARD_WRONG_DECISION = -2.0
REWARD_CORRECT_EXTRACTION = 0.5
REWARD_WRONG_EXTRACTION = -0.2
REWARD_PO_MATCH_CORRECT = 1.0
REWARD_PO_MATCH_WRONG = -0.5
REWARD_REDUNDANT_ACTION = -0.1
REWARD_STEP_COST = -0.05  # small penalty per step to encourage efficiency


def _compute_task_max_reward(task: TaskDefinition) -> float:
    """Compute the theoretical maximum reward for a task.

    Only counts rewards for the _mandatory_ actions an optimal agent
    would take. This ensures a perfect run normalizes to exactly 1.0.
    """
    max_r = 0.0
    required = task.required_subtasks

    if "categorized" in required:
        max_r += REWARD_CORRECT_CATEGORY
    if "priority_set" in required:
        max_r += REWARD_CORRECT_PRIORITY
    if "po_validated" in required:
        max_r += REWARD_PO_MATCH_CORRECT

    # Each expected issue flagged correctly
    max_r += len(task.expected_issues) * REWARD_CORRECT_ISSUE_FLAG

    # Correct decision bonus + all-subtasks-done bonus
    max_r += REWARD_CORRECT_DECISION
    max_r += 1.0  # completion bonus

    return max_r


class InvoiceTriageEnvironment(
    Environment[InvoiceAction, InvoiceObservation, InvoiceState]
):
    """An accounts-payable invoice triage environment.

    The agent receives invoices and must:
    1. Categorize the spend
    2. Set processing priority
    3. Validate against purchase orders
    4. Flag any issues (amount mismatches, duplicates, tax errors, etc.)
    5. Make a final approve / reject / escalate decision

    Rewards are shaped to teach thorough, accurate processing.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._task_id = task_id
        self._difficulty = difficulty
        self._task: Optional[TaskDefinition] = None
        self._state = InvoiceState()
        self._completed_subtasks: List[str] = []
        self._issues_found: List[str] = []
        self._extractions: Dict[str, str] = {}
        self._final_decision: Optional[str] = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False

    # -------------------------------------------------------------------------
    # Environment interface
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Reset the environment with a new invoice processing task."""
        self._reset_rubric()

        if seed is not None:
            random.seed(seed)

        # Select task
        self._task = self._select_task()

        # Reset all state
        self._completed_subtasks = []
        self._issues_found = []
        self._extractions = {}
        self._final_decision = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False

        self._state = InvoiceState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            task_difficulty=self._task.difficulty,
            completed_subtasks=[],
            required_subtasks=list(self._task.required_subtasks),
            issues_found=[],
            issues_expected=list(self._task.expected_issues),
            final_decision=None,
            expected_decision=self._task.expected_decision,
            cumulative_reward=0.0,
        )

        return self._build_observation(
            feedback="New invoice loaded. Begin processing.",
            error=None,
        )

    def step(
        self,
        action: InvoiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Execute one processing step."""
        if self._done:
            return self._build_observation(
                feedback="Episode is already complete. Call reset() to start a new one.",
                error="Episode finished.",
            )

        if self._task is None:
            return self._build_observation(
                feedback="",
                error="No task loaded. Call reset() first.",
            )

        self._step_count += 1
        self._state.step_count = self._step_count

        # Apply step cost
        self._cumulative_reward += REWARD_STEP_COST

        # Dispatch to the correct handler
        reward, feedback, error = self._handle_action(action)
        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward

        # Check termination conditions
        if self._step_count >= self._task.max_steps and not self._done:
            self._done = True
            # Penalise for running out of steps without deciding
            if self._final_decision is None:
                self._cumulative_reward += REWARD_WRONG_DECISION
                feedback += " | Episode timed out without a final decision."
            self._state.cumulative_reward = self._cumulative_reward

        obs = self._build_observation(feedback=feedback, error=error)
        obs = self._apply_transform(obs)
        return obs

    @property
    def state(self) -> InvoiceState:
        """Return current environment state."""
        return self._state

    # -------------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------------

    def _handle_action(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        """Route action to the correct handler. Returns (reward, feedback, error)."""
        handlers = {
            ActionType.CATEGORIZE: self._handle_categorize,
            ActionType.SET_PRIORITY: self._handle_set_priority,
            ActionType.FLAG_ISSUE: self._handle_flag_issue,
            ActionType.APPROVE: self._handle_approve,
            ActionType.REJECT: self._handle_reject,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.EXTRACT_FIELD: self._handle_extract_field,
            ActionType.VALIDATE_MATCH: self._handle_validate_match,
            ActionType.SUBMIT_DECISION: self._handle_submit_decision,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return 0.0, "", f"Unknown action type: {action.action_type}"
        return handler(action)

    def _handle_categorize(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        if action.category is None:
            return 0.0, "", "CATEGORIZE requires 'category' field."

        if "categorized" in self._completed_subtasks:
            return REWARD_REDUNDANT_ACTION, "Already categorized.", None

        assert self._task is not None
        self._completed_subtasks.append("categorized")
        self._state.completed_subtasks = list(self._completed_subtasks)

        if action.category.value == self._task.expected_category:
            return (
                REWARD_CORRECT_CATEGORY,
                f"Correct! Category set to '{action.category.value}'.",
                None,
            )
        return (
            REWARD_WRONG_CATEGORY,
            f"Incorrect category '{action.category.value}'. "
            f"Expected '{self._task.expected_category}'.",
            None,
        )

    def _handle_set_priority(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        if action.priority is None:
            return 0.0, "", "SET_PRIORITY requires 'priority' field."

        if "priority_set" in self._completed_subtasks:
            return REWARD_REDUNDANT_ACTION, "Priority already set.", None

        assert self._task is not None
        self._completed_subtasks.append("priority_set")
        self._state.completed_subtasks = list(self._completed_subtasks)

        if action.priority.value == self._task.expected_priority:
            return (
                REWARD_CORRECT_PRIORITY,
                f"Correct! Priority set to '{action.priority.value}'.",
                None,
            )
        return (
            REWARD_WRONG_PRIORITY,
            f"Priority set to '{action.priority.value}'. "
            f"Expected '{self._task.expected_priority}'.",
            None,
        )

    def _handle_flag_issue(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        if action.issue_type is None:
            return 0.0, "", "FLAG_ISSUE requires 'issue_type' field."

        assert self._task is not None

        if "issue_flagged" not in self._completed_subtasks:
            self._completed_subtasks.append("issue_flagged")
            self._state.completed_subtasks = list(self._completed_subtasks)

        issue = action.issue_type.value
        if issue in self._issues_found:
            return (
                REWARD_REDUNDANT_ACTION,
                f"Issue '{issue}' already flagged.",
                None,
            )

        self._issues_found.append(issue)
        self._state.issues_found = list(self._issues_found)

        if issue in self._task.expected_issues:
            desc = action.issue_description or ""
            return (
                REWARD_CORRECT_ISSUE_FLAG,
                f"Good catch! '{issue}' flagged correctly. {desc}",
                None,
            )
        return (
            REWARD_FALSE_POSITIVE_ISSUE,
            f"False positive: '{issue}' is not a real issue here.",
            None,
        )

    def _handle_extract_field(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        if action.field_name is None or action.field_value is None:
            return (
                0.0,
                "",
                "EXTRACT_FIELD requires 'field_name' and 'field_value'.",
            )

        assert self._task is not None
        key = action.field_name.lower().strip()
        val = action.field_value.strip()
        self._extractions[key] = val

        expected = self._task.expected_extractions.get(key)
        if expected is not None and val == expected:
            return (
                REWARD_CORRECT_EXTRACTION,
                f"Correct extraction: {key} = {val}",
                None,
            )
        if expected is not None:
            return (
                REWARD_WRONG_EXTRACTION,
                f"Extraction mismatch for '{key}': got '{val}', expected '{expected}'.",
                None,
            )
        return 0.0, f"Extracted {key} = {val} (no ground truth).", None

    def _handle_validate_match(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        if action.match_result is None:
            return (
                0.0,
                "",
                "VALIDATE_MATCH requires 'match_result' (bool).",
            )

        if "po_validated" in self._completed_subtasks:
            return REWARD_REDUNDANT_ACTION, "PO already validated.", None

        assert self._task is not None
        self._completed_subtasks.append("po_validated")
        self._state.completed_subtasks = list(self._completed_subtasks)

        has_issues = len(self._task.expected_issues) > 0
        po_should_match = not has_issues or "amount_mismatch" not in self._task.expected_issues  # noqa: E501

        if action.match_result == po_should_match:
            result_str = "matches" if action.match_result else "does NOT match"
            return (
                REWARD_PO_MATCH_CORRECT,
                f"Correct: invoice {result_str} the purchase order.",
                None,
            )
        return (
            REWARD_PO_MATCH_WRONG,
            "Incorrect PO validation result.",
            None,
        )

    def _handle_approve(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        return self._finalize_decision("approve", action.reason)

    def _handle_reject(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        return self._finalize_decision("reject", action.reason)

    def _handle_escalate(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        return self._finalize_decision("escalate", action.reason)

    def _handle_submit_decision(
        self, action: InvoiceAction
    ) -> tuple[float, str, Optional[str]]:
        # submit_decision uses the reason field to encode the decision
        reason = (action.reason or "").strip().lower()
        if reason in ("approve", "reject", "escalate"):
            return self._finalize_decision(reason, action.issue_description)
        return 0.0, "", "SUBMIT_DECISION requires 'reason' to be approve/reject/escalate."

    def _finalize_decision(
        self, decision: str, reason: Optional[str]
    ) -> tuple[float, str, Optional[str]]:
        """Finalise the episode with the agent's decision."""
        if self._final_decision is not None:
            return REWARD_REDUNDANT_ACTION, "Decision already submitted.", None

        assert self._task is not None
        self._final_decision = decision
        self._done = True
        self._state.final_decision = decision

        if "decision_made" not in self._completed_subtasks:
            self._completed_subtasks.append("decision_made")
            self._state.completed_subtasks = list(self._completed_subtasks)

        reward = 0.0

        # Correct decision?
        if decision == self._task.expected_decision:
            reward += REWARD_CORRECT_DECISION
            fb = f"Decision '{decision}' is CORRECT."
        else:
            reward += REWARD_WRONG_DECISION
            fb = (
                f"Decision '{decision}' is INCORRECT. "
                f"Expected '{self._task.expected_decision}'."
            )

        # Penalise missed issues
        missed = set(self._task.expected_issues) - set(self._issues_found)
        if missed:
            missed_penalty = len(missed) * REWARD_MISSED_ISSUE_PENALTY
            reward += missed_penalty
            fb += f" Missed issues: {', '.join(sorted(missed))}."

        # Bonus for completing all required subtasks
        completed_all = all(
            s in self._completed_subtasks for s in self._task.required_subtasks
        )
        if completed_all:
            reward += 1.0
            fb += " All required subtasks completed — bonus awarded."

        if reason:
            fb += f" Reason: {reason}"

        return reward, fb, None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _select_task(self) -> TaskDefinition:
        """Select a task based on config or randomly."""
        if self._task_id and self._task_id in ALL_TASKS:
            return ALL_TASKS[self._task_id]

        if self._difficulty and self._difficulty in TASKS_BY_DIFFICULTY:
            pool = TASKS_BY_DIFFICULTY[self._difficulty]
        else:
            pool = list(ALL_TASKS.values())

        return random.choice(pool)

    def _build_observation(
        self,
        feedback: str,
        error: Optional[str],
    ) -> InvoiceObservation:
        """Build the observation to send to the agent."""
        assert self._task is not None or error is not None

        available_actions = self._available_actions()

        # Normalize reward to 0.0–1.0 (required by OpenEnv grading spec)
        # Step costs are a training signal for efficiency, not a correctness
        # penalty — so we normalize on the _correctness_ component only.
        if self._task is not None:
            max_reward = _compute_task_max_reward(self._task)
            correctness_reward = self._cumulative_reward - (self._step_count * REWARD_STEP_COST)
            normalized_reward = max(0.0, min(1.0, correctness_reward / max_reward)) if max_reward > 0 else 0.0
        else:
            normalized_reward = 0.0

        return InvoiceObservation(
            done=self._done,
            reward=round(normalized_reward, 4),
            goal=self._task.goal if self._task else "",
            invoice=self._task.invoice if self._task else None,
            purchase_order=self._task.purchase_order if self._task else None,
            historical_invoices=(
                self._task.historical_invoices if self._task else []
            ),
            available_actions=available_actions,
            last_action_feedback=feedback,
            last_action_error=error,
            progress={
                s: (s in self._completed_subtasks)
                for s in (self._task.required_subtasks if self._task else [])
            },
            step_number=self._step_count,
            max_steps=self._task.max_steps if self._task else 15,
        )

    def _available_actions(self) -> List[str]:
        """Return list of actions the agent can still take."""
        if self._done:
            return []

        actions = []
        if "categorized" not in self._completed_subtasks:
            actions.append("categorize")
        if "priority_set" not in self._completed_subtasks:
            actions.append("set_priority")
        if "po_validated" not in self._completed_subtasks and (
            self._task and self._task.purchase_order is not None
        ):
            actions.append("validate_match")

        # Always available until done
        actions.extend(["flag_issue", "extract_field"])

        # Terminal actions
        actions.extend(["approve", "reject", "escalate", "submit_decision"])

        return actions
