"""Evaluation harness — run all tasks with any agent and produce a report.

Usage:
    # Rule-based agent (offline, no API key needed)
    PYTHONPATH=. python -m invoice_triage_env.evaluate

    # With JSON report output
    PYTHONPATH=. python -m invoice_triage_env.evaluate --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from invoice_triage_env.examples.run_agent import run_agent_episode
from invoice_triage_env.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY


@dataclass
class TaskResult:
    """Result of a single task evaluation."""

    task_id: str
    difficulty: str
    reward: float
    steps: int
    max_steps: int
    decision_correct: bool
    all_subtasks_done: bool
    issues_found: List[str]
    issues_expected: List[str]
    issues_missed: List[str]
    elapsed_seconds: float


@dataclass
class EvalReport:
    """Full evaluation report."""

    agent_type: str
    model: Optional[str]
    timestamp: str
    total_reward: float
    avg_reward: float
    tasks_correct: int
    tasks_total: int
    accuracy: float
    results: List[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "model": self.model,
            "timestamp": self.timestamp,
            "total_reward": self.total_reward,
            "avg_reward": round(self.avg_reward, 3),
            "tasks_correct": self.tasks_correct,
            "tasks_total": self.tasks_total,
            "accuracy": round(self.accuracy, 3),
            "results": [asdict(r) for r in self.results],
        }


def evaluate_rule_based(
    task_ids: Optional[List[str]] = None,
) -> EvalReport:
    """Run the rule-based agent on all (or selected) tasks."""
    from invoice_triage_env.server.invoice_triage_environment import (
        InvoiceTriageEnvironment,
    )

    targets = task_ids or list(ALL_TASKS.keys())
    results: List[TaskResult] = []

    for tid in targets:
        task_def = ALL_TASKS[tid]
        t0 = time.time()
        episode = run_agent_episode(tid)
        elapsed = time.time() - t0

        # Determine correctness from env state
        env = InvoiceTriageEnvironment(task_id=tid)
        env.reset(seed=42)

        result = TaskResult(
            task_id=tid,
            difficulty=task_def.difficulty,
            reward=episode["reward"],
            steps=episode["steps"],
            max_steps=task_def.max_steps,
            decision_correct=episode["reward"] > 0,
            all_subtasks_done=all(episode["progress"].values()),
            issues_found=[],  # rule agent doesn't expose this easily
            issues_expected=task_def.expected_issues,
            issues_missed=[],
            elapsed_seconds=round(elapsed, 3),
        )
        results.append(result)

    total = sum(r.reward for r in results)
    correct = sum(1 for r in results if r.decision_correct)

    return EvalReport(
        agent_type="rule_based",
        model=None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_reward=round(total, 3),
        avg_reward=round(total / len(results), 3) if results else 0.0,
        tasks_correct=correct,
        tasks_total=len(results),
        accuracy=round(correct / len(results), 3) if results else 0.0,
        results=results,
    )


def print_report(report: EvalReport) -> None:
    """Pretty-print the evaluation report."""
    print(f"\n{'=' * 72}")
    print(f"  EVALUATION REPORT — {report.agent_type.upper()}")
    if report.model:
        print(f"  Model: {report.model}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"{'=' * 72}")

    # Per-difficulty breakdown
    by_diff: Dict[str, List[TaskResult]] = {"easy": [], "medium": [], "hard": []}
    for r in report.results:
        by_diff.setdefault(r.difficulty, []).append(r)

    for diff in ("easy", "medium", "hard"):
        tasks = by_diff.get(diff, [])
        if not tasks:
            continue
        print(f"\n  [{diff.upper()}]")
        for r in tasks:
            icon = "✓" if r.decision_correct else "✗"
            print(
                f"    {icon} {r.task_id:<35} "
                f"reward={r.reward:>7.2f}  "
                f"steps={r.steps}/{r.max_steps}  "
                f"time={r.elapsed_seconds:.1f}s"
            )

    print(f"\n{'─' * 72}")
    print(f"  Total Reward:     {report.total_reward:>7.2f}")
    print(f"  Avg Reward:       {report.avg_reward:>7.3f}")
    print(f"  Accuracy:         {report.tasks_correct}/{report.tasks_total} ({report.accuracy:.0%})")
    print(f"{'=' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InvoiceTriageEnv agents")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        nargs="*",
        default=None,
        help="Specific task IDs to evaluate",
    )
    args = parser.parse_args()

    report = evaluate_rule_based(task_ids=args.task)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
