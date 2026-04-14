"""REINFORCE training loop for InvoiceTriageEnv.

Algorithm: REINFORCE with baseline (actor-critic variant).
  - Runs full episodes in the environment
  - Collects (log_prob, reward, value) per step
  - Computes discounted returns with a value baseline
  - Updates policy via policy gradient + value loss

Usage::

    python -m invoice_triage_env.training.train_reinforce \\
        --episodes 500 \\
        --tasks easy_approve_clean easy_reject_no_po \\  # or 'all'
        --save-dir outputs/checkpoints \\
        --plot-dir outputs/

Output:
    outputs/checkpoints/policy_ep{N}.pt   — saved checkpoints
    outputs/training_curves.png            — learning curve plot
    outputs/benchmark_results.json         — final benchmark scores
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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
from invoice_triage_env.training.obs_encoder import ACTION_NAMES, ObservationEncoder
from invoice_triage_env.training.policy import ActorCriticPolicy

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

GAMMA = 0.99           # discount factor
LR = 2e-4              # AdamW learning rate (slightly lower for stability)
ENTROPY_COEF = 0.05    # entropy bonus
VALUE_COEF = 0.5       # value loss weight
MAX_GRAD_NORM = 1.0    # gradient clipping
CHECKPOINT_EVERY = 500 # save policy every N episodes
BATCH_SIZE = 16        # episodes per gradient update (batch advantage normalization)


# ---------------------------------------------------------------------------
# Action construction helpers
# ---------------------------------------------------------------------------

# Maps action index → ActionType
IDX_TO_ACTION: Dict[int, ActionType] = {
    i: ActionType(name) for i, name in enumerate(ACTION_NAMES)
}

# Canonical category/priority/issue sequences (fixed order for argmax)
_CATEGORIES = list(InvoiceCategory)
_PRIORITIES = list(Priority)
_ISSUES = list(IssueType)


# Keyword → expected number of issues hinted in the goal string
_ISSUE_KEYWORDS = [
    "mismatch", "duplicate", "missing po", "no po", "missing approval",
    "suspicious", "tax error", "over budget", "budget", "fraud",
    "ghost", "split", "currency", "forex", "overbill", "abuse",
    "date anomaly", "tax",
]


def _count_issue_hints(goal: str) -> int:
    """Count how many distinct issue keywords appear in the goal string."""
    g = goal.lower()
    return sum(1 for kw in _ISSUE_KEYWORDS if kw in g)


def _next_required_action(obs: InvoiceObservation) -> Optional[str]:
    """Determine the next mandatory subtask that has not yet been completed.

    Required ordering: categorize → set_priority → [validate_match] → flag_issue → decide.

    For tasks with issue hints in the goal, flag_issue is forced once before
    any terminal action is allowed. The curriculum then lets the policy decide
    the final approve/reject/escalate.

    Returns None when all prerequisites are satisfied — lets policy decide terminal.
    """
    prog = obs.progress
    avail = set(obs.available_actions)
    goal = obs.goal

    if not prog.get("categorized") and "categorize" in avail:
        return "categorize"
    if not prog.get("priority_set") and "set_priority" in avail:
        return "set_priority"
    # Validate PO if the action is available and not yet done
    if not prog.get("po_validated") and "validate_match" in avail:
        return "validate_match"

    # Force flag_issue at least once for any task that has issue hints
    n_hints = _count_issue_hints(goal)
    if n_hints > 0 and not prog.get("issue_flagged") and "flag_issue" in avail:
        return "flag_issue"

    return None  # let the policy choose the terminal action



def _build_action(action_idx: int, obs: InvoiceObservation) -> InvoiceAction:
    """Convert action index to a concrete InvoiceAction with sensible payload.

    If the policy tries a terminal action before completing required subtasks,
    we redirect to the next mandatory step (curriculum forcing).
    This dramatically improves early-episode learning signal.
    """
    action_type = IDX_TO_ACTION[action_idx]
    goal_lower = (obs.goal + obs.last_action_feedback).lower()

    # ---- Curriculum forcing: override terminal actions before prerequisites ----
    TERMINAL_ACTIONS = {ActionType.APPROVE, ActionType.REJECT, ActionType.ESCALATE,
                        ActionType.SUBMIT_DECISION}
    if action_type in TERMINAL_ACTIONS:
        forced = _next_required_action(obs)
        if forced is not None:
            action_type = ActionType(forced)  # redirect to required step

    # ---- Categorize --------------------------------------------------------
    if action_type == ActionType.CATEGORIZE:
        kw_map = {
            "software": InvoiceCategory.SOFTWARE,
            "license": InvoiceCategory.SOFTWARE,
            "consulting": InvoiceCategory.CONSULTING,
            "marketing": InvoiceCategory.MARKETING,
            "supplies": InvoiceCategory.SUPPLIES,
            "maintenance": InvoiceCategory.MAINTENANCE,
            "equipment": InvoiceCategory.EQUIPMENT,
            "travel": InvoiceCategory.TRAVEL,
            "utilities": InvoiceCategory.UTILITIES,
        }
        cat = InvoiceCategory.OTHER
        for kw, c in kw_map.items():
            if kw in goal_lower:
                cat = c
                break
        if cat == InvoiceCategory.OTHER and obs.invoice:
            vendor = obs.invoice.vendor_name.lower()
            for kw, c in kw_map.items():
                if kw in vendor:
                    cat = c
                    break
        return InvoiceAction(action_type=action_type, category=cat)

    # ---- Set Priority -------------------------------------------------------
    elif action_type == ActionType.SET_PRIORITY:
        amount = obs.invoice.total_amount if obs.invoice else 0
        if amount > 50_000 or "urgent" in goal_lower or "fraud" in goal_lower:
            priority = Priority.URGENT
        elif amount > 10_000 or "high" in goal_lower or "suspicious" in goal_lower or "mismatch" in goal_lower:
            priority = Priority.HIGH
        elif amount > 1_000 or "medium" in goal_lower or "duplicate" in goal_lower:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW
        return InvoiceAction(action_type=action_type, priority=priority)

    # ---- Flag Issue — cycles through ALL matching issues per call ----------
    elif action_type == ActionType.FLAG_ISSUE:
        # Ordered priority list: more specific checks first
        kw_issue_list = [
            ("ghost",            IssueType.SUSPICIOUS_VENDOR),
            ("phantom",          IssueType.SUSPICIOUS_VENDOR),
            ("suspicious",       IssueType.SUSPICIOUS_VENDOR),
            ("fraud",            IssueType.SUSPICIOUS_VENDOR),
            ("duplicate",        IssueType.DUPLICATE_INVOICE),
            ("missing po",       IssueType.MISSING_PO),
            ("no po",            IssueType.MISSING_PO),
            ("purchase order",   IssueType.MISSING_PO),
            ("missing approval", IssueType.MISSING_APPROVAL),
            ("approval",         IssueType.MISSING_APPROVAL),
            ("tax",              IssueType.TAX_ERROR),
            ("over budget",      IssueType.OVER_BUDGET),
            ("budget",           IssueType.OVER_BUDGET),
            ("mismatch",         IssueType.AMOUNT_MISMATCH),
            ("currency",         IssueType.AMOUNT_MISMATCH),
            ("forex",            IssueType.AMOUNT_MISMATCH),
            ("overbill",         IssueType.AMOUNT_MISMATCH),
            ("date anomaly",     IssueType.DATE_ANOMALY),
            ("vendor",           IssueType.VENDOR_MISMATCH),
        ]
        # Build the list of matching issues in order
        matched_issues = []
        seen: set = set()
        for kw, iss in kw_issue_list:
            if kw in goal_lower and iss not in seen:
                matched_issues.append(iss)
                seen.add(iss)

        # Cycle through issues using step_number as index (each flag_issue call advances step)
        flags_so_far = max(0, obs.step_number - 3)  # approximate: categorize+priority+validate = 3 preceding steps
        if matched_issues:
            issue = matched_issues[flags_so_far % len(matched_issues)]
        else:
            issue = IssueType.AMOUNT_MISMATCH  # safe default

        return InvoiceAction(
            action_type=action_type,
            issue_type=issue,
            issue_description=f"Policy detected [{flags_so_far + 1}]: {issue.value}",
        )

    # ---- Extract Field -----------------------------------------------------
    elif action_type == ActionType.EXTRACT_FIELD:
        inv = obs.invoice
        if inv:
            return InvoiceAction(
                action_type=action_type,
                field_name="total_amount",
                field_value=str(inv.total_amount),
            )
        return InvoiceAction(
            action_type=action_type,
            field_name="vendor_name",
            field_value="unknown",
        )

    # ---- Validate Match ----------------------------------------------------
    elif action_type == ActionType.VALIDATE_MATCH:
        match = True
        if obs.invoice and obs.purchase_order:
            diff = abs(obs.invoice.total_amount - obs.purchase_order.total_amount)
            match = diff < 0.01
        return InvoiceAction(action_type=action_type, match_result=match)

    # ---- Terminal decisions ------------------------------------------------
    elif action_type == ActionType.APPROVE:
        return InvoiceAction(action_type=action_type, reason="Policy decision: approve")
    elif action_type == ActionType.REJECT:
        return InvoiceAction(action_type=action_type, reason="Policy decision: reject")
    elif action_type == ActionType.ESCALATE:
        return InvoiceAction(action_type=action_type, reason="Policy decision: escalate")
    elif action_type == ActionType.SUBMIT_DECISION:
        return InvoiceAction(action_type=action_type, reason="approve")

    return InvoiceAction(action_type=ActionType.APPROVE, reason="Fallback")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: InvoiceTriageEnvironment,
    policy: ActorCriticPolicy,
    encoder: ObservationEncoder,
    task_id: Optional[str] = None,
    deterministic: bool = False,
    seed: int = 0,
) -> dict:
    """Run one episode. Returns trajectory data for policy update.

    Curriculum forcing is always active: before a terminal decision action
    is allowed, all required intermediate steps run first. In deterministic
    eval mode the policy picks ONLY the final decision (approve/reject/escalate)
    while the sequencer handles every preceding step automatically.
    """
    obs = env.reset(seed=seed)

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    entropies: List[torch.Tensor] = []
    forced_masks: List[bool] = []

    prev_reward = 0.0
    from torch.distributions import Categorical

    while not obs.done:
        obs_tensor = encoder.encode(obs).unsqueeze(0).to(device)  # (1, OBS_DIM)
        logits, value_raw = policy(obs_tensor)
        dist = Categorical(logits=logits)
        value = value_raw.squeeze()

        # --- Check if curriculum sequencer mandates the next action ----------
        forced_name = _next_required_action(obs)
        avail_set = set(obs.available_actions)

        if forced_name is not None and forced_name in avail_set:
            # Override to the mandatory next step
            action_idx = ACTION_NAMES.index(forced_name)
            is_forced = True
        else:
            # Let the policy decide (terminal or optional actions)
            is_forced = False
            if deterministic:
                # Mask unavailable actions to -inf so argmax picks the best
                # *available* action, not just the globally highest raw logit.
                # Without this, hard-task REJECT bias makes greedy always pick
                # REJECT even for approve tasks where APPROVE has a higher logit
                # among the available terminal actions.
                masked_logits = logits.squeeze(0).clone()
                for i, aname in enumerate(ACTION_NAMES):
                    if aname not in avail_set:
                        masked_logits[i] = float("-inf")
                action_idx = int(masked_logits.argmax(dim=-1).item())
            else:
                action_idx = int(dist.sample().item())

            # If policy picks an unavailable action (shouldn't happen with masking),
            # fall back to random available as a safety net.
            if ACTION_NAMES[action_idx] not in avail_set:
                available_idxs = [i for i, n in enumerate(ACTION_NAMES) if n in avail_set]
                action_idx = random.choice(available_idxs) if available_idxs else action_idx

        action_tensor = torch.tensor([action_idx], device=device)
        log_prob = dist.log_prob(action_tensor)

        action = _build_action(action_idx, obs)
        obs = env.step(action)

        step_reward = obs.reward - prev_reward  # delta reward
        prev_reward = obs.reward

        log_probs.append(log_prob.squeeze())
        values.append(value)
        rewards.append(step_reward)
        entropies.append(dist.entropy().squeeze())
        forced_masks.append(is_forced)

    return {
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards,
        "entropies": entropies,
        "forced_masks": forced_masks,
        "final_score": obs.reward,
        "steps": len(rewards),
    }


# ---------------------------------------------------------------------------
# Returns computation
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float = GAMMA) -> torch.Tensor:
    """Compute discounted returns, normalised for stability."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)
    # Removing per-episode return normalization.
    # Normalizing within a single episode distorts the absolute returns, causing
    # terrible terminal actions to have positive returns relative to the episode mean!
    return ret


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _sample_task(use_tasks: List[str], episode: int, total_episodes: int) -> str:
    """Curriculum-aware task sampling.

    Weights shift from easy-heavy → balanced → hard-heavy as training
    progresses. Tasks without 'easy'/'medium'/'hard' prefix are treated
    as medium difficulty.
    """
    progress = episode / max(total_episodes, 1)  # 0.0 → 1.0

    # Difficulty weight schedules
    if progress < 0.3:
        w_easy, w_med, w_hard = 0.55, 0.30, 0.15
    elif progress < 0.6:
        w_easy, w_med, w_hard = 0.30, 0.40, 0.30
    else:
        w_easy, w_med, w_hard = 0.15, 0.35, 0.50

    def difficulty(tid: str) -> str:
        if tid.startswith("easy"):
            return "easy"
        if tid.startswith("hard"):
            return "hard"
        return "medium"

    weights = [
        w_easy if difficulty(t) == "easy" else
        w_med  if difficulty(t) == "medium" else
        w_hard
        for t in use_tasks
    ]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return random.choices(use_tasks, weights=weights, k=1)[0]


def train(
    n_episodes: int = 500,
    task_ids: Optional[List[str]] = None,
    save_dir: str = "outputs/checkpoints",
    plot_dir: str = "outputs",
    seed: int = 42,
    resume_checkpoint: Optional[str] = None,
) -> dict:
    """Main REINFORCE training loop with batch advantage normalization.

    Args:
        n_episodes: Total training episodes.
        task_ids: List of task IDs to train on. None = all tasks.
        save_dir: Directory to save policy checkpoints.
        plot_dir: Directory to save training curve plots.
        seed: Random seed for reproducibility.
        resume_checkpoint: Path to a .pt checkpoint to warm-start from.

    Returns:
        dict with training history and final benchmark scores.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    encoder = ObservationEncoder()
    policy = ActorCriticPolicy().to(device)
    optimizer = AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    # Scheduler: warm up for 5% of episodes, then cosine anneal
    scheduler = CosineAnnealingLR(optimizer, T_max=n_episodes, eta_min=LR * 0.05)

    start_episode = 1
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        policy.load_state_dict(ckpt["model_state"])
        # Don't restore optimizer — fresh optimizer is better when resuming
        start_episode = ckpt.get("episode", 1) + 1
        print(f"  ✓ Resumed from checkpoint: {resume_checkpoint} (ep {start_episode - 1})")

    use_tasks = task_ids or list(ALL_TASKS.keys())
    print(f"\n{'='*65}")
    print(f"  InvoiceTriageEnv — REINFORCE Training (batch advantage normalization)")
    print(f"  Device: {device} | Tasks: {len(use_tasks)} | Episodes: {n_episodes}")
    print(f"  LR: {LR} | γ: {GAMMA} | Batch: {BATCH_SIZE} | Entropy: {ENTROPY_COEF}")
    print(f"  Policy params: {policy.num_parameters:,}")
    print(f"{'='*65}\n")

    history: Dict[str, List] = {
        "episode": [],
        "score": [],
        "loss": [],
        "steps": [],
        "task": [],
    }

    window = []  # rolling average window (last 20)
    batch_scores: List[float] = []

    # Epoch loop — each epoch = BATCH_SIZE episodes → 1 gradient update
    n_batches = (n_episodes + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE + 1
        batch_end   = min(batch_start + BATCH_SIZE - 1, n_episodes)
        ep_range    = range(batch_start, batch_end + 1)

        # ── 1. Collect trajectories ──────────────────────────────────────────
        policy.train()
        batch_trajs: list = []
        batch_rets:  list = []

        for ep in ep_range:
            task_id = _sample_task(use_tasks, ep, n_episodes)
            env = InvoiceTriageEnvironment(task_id=task_id)
            traj = run_episode(env, policy, encoder, task_id, seed=ep + seed)
            returns = compute_returns(traj["rewards"])
            traj["task_id"] = task_id
            batch_trajs.append(traj)
            batch_rets.append(returns)

            score = traj["final_score"]
            window.append(score)
            if len(window) > 20:
                window.pop(0)
            history["episode"].append(ep)
            history["score"].append(score)
            history["steps"].append(traj["steps"])
            history["task"].append(task_id)

        # ── 2. Batch advantage normalization ─────────────────────────────────
        # Collect all raw advantages across the batch, then normalize to
        # mean=0 std=1  →  consistent gradient magnitude regardless of reward scale.
        all_raw_adv: list = []
        batch_advantages: list = []
        for traj, returns in zip(batch_trajs, batch_rets):
            values_stack = torch.stack(traj["values"]).detach()
            raw_adv = returns - values_stack
            batch_advantages.append(raw_adv)
            all_raw_adv.append(raw_adv)

        adv_cat  = torch.cat(all_raw_adv)   # all steps in this batch
        adv_mean = adv_cat.mean()
        adv_std  = adv_cat.std() + 1e-8     # avoid division by zero

        # ── 3. Policy update ─────────────────────────────────────────────────
        optimizer.zero_grad()
        total_policy_loss  = torch.tensor(0.0, device=device)
        total_value_loss   = torch.tensor(0.0, device=device)
        total_entropy_bonus= torch.tensor(0.0, device=device)
        total_steps = 0

        for traj, returns, raw_adv in zip(batch_trajs, batch_rets, batch_advantages):
            norm_adv = (raw_adv - adv_mean) / adv_std  # batch-normalised
            for i, (lp, val, ret, ent, forced) in enumerate(zip(
                traj["log_probs"], traj["values"], returns,
                traj["entropies"], traj["forced_masks"]
            )):
                adv = norm_adv[i]
                if forced:
                    total_policy_loss = total_policy_loss - lp   # BC for forced steps
                else:
                    total_policy_loss = total_policy_loss - lp * adv
                total_value_loss   = total_value_loss   + F.mse_loss(val.unsqueeze(0), ret.unsqueeze(0))
                total_entropy_bonus= total_entropy_bonus - ent.mean()
            total_steps += max(len(traj["log_probs"]), 1)

        loss_val = None
        if total_steps > 0:
            total_loss = (
                total_policy_loss  / total_steps
                + VALUE_COEF * total_value_loss   / total_steps
                + ENTROPY_COEF * total_entropy_bonus / total_steps
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            loss_val = total_loss.item()

        # ── 4. Logging ───────────────────────────────────────────────────────
        avg  = sum(window) / len(window)
        ep   = batch_end
        score = batch_trajs[-1]["final_score"] if batch_trajs else 0.0
        task_id = batch_trajs[-1]["task_id"]   if batch_trajs else ""
        history["loss"].extend([loss_val] * len(ep_range))

        # Print every batch
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        if batch_idx % max(1, n_batches // 50) == 0 or ep == n_episodes:
            print(
                f"  Ep {ep:5d}/{n_episodes} | "
                f"task={task_id:<35} | "
                f"score={score:.4f} | "
                f"avg20={avg:.4f} | "
                f"loss={loss_str}"
            )

        # ---- Checkpointing -------------------------------------------------
        if ep % CHECKPOINT_EVERY == 0 or ep == n_episodes:
            ckpt_path = Path(save_dir) / f"policy_ep{ep:05d}.pt"
            torch.save(
                {
                    "episode": ep,
                    "model_state": policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "avg_score": avg,
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    # ---- Final benchmark ---------------------------------------------------
    # 1 deterministic trial + 4 stochastic trials → mean of all 5.
    # Deterministic greedy removes sampling variance so we can see true
    # policy capability. Stochastic trials catch cases where greedy is
    # suboptimal (entropy > 0 can help).
    print(f"\n{'='*65}")
    print("  Running final benchmark (1 greedy + 4 stochastic per task)...")
    benchmark = {}
    policy.eval()
    with torch.no_grad():
        for tid in sorted(ALL_TASKS.keys()):
            task_scores = []
            # One deterministic (greedy) run
            env = InvoiceTriageEnvironment(task_id=tid)
            traj = run_episode(env, policy, encoder, tid, deterministic=True, seed=0)
            task_scores.append(traj["final_score"])
            best_steps = traj["steps"]
            # Four stochastic runs
            for trial_seed in range(1, 5):
                env = InvoiceTriageEnvironment(task_id=tid)
                traj = run_episode(env, policy, encoder, tid, deterministic=False, seed=trial_seed)
                task_scores.append(traj["final_score"])
            avg_score = sum(task_scores) / len(task_scores)
            greedy_score = task_scores[0]
            status = "✅" if max(greedy_score, avg_score) >= 0.5 else "❌"
            benchmark[tid] = {
                "score": avg_score,
                "greedy_score": greedy_score,
                "steps": best_steps,
                "trials": task_scores,
            }
            print(f"    {status} {tid:<38} greedy={greedy_score:.4f}  avg={avg_score:.4f}")

    avg_benchmark   = sum(v["score"]        for v in benchmark.values()) / len(benchmark)
    avg_greedy      = sum(v["greedy_score"] for v in benchmark.values()) / len(benchmark)
    n_passing_greedy = sum(1 for v in benchmark.values() if v["greedy_score"] >= 0.5)
    n_passing_avg    = sum(1 for v in benchmark.values() if v["score"]        >= 0.5)
    print(f"\n  Greedy avg:  {avg_greedy:.4f}  |  Stochastic avg: {avg_benchmark:.4f}")
    print(f"  Passing (greedy ≥0.5): {n_passing_greedy}/{len(benchmark)}   "
          f"Passing (avg ≥0.5): {n_passing_avg}/{len(benchmark)}")
    print(f"{'='*65}\n")

    # ---- Save results -------------------------------------------------------
    # When running --benchmark-only (n_episodes=0), use checkpoint name as label
    ep_label: int | str = n_episodes
    if n_episodes == 0 and resume_checkpoint:
        ep_label = Path(resume_checkpoint).stem  # e.g. "policy_ep15000"
    results = {
        "history": history,
        "benchmark": benchmark,
        "avg_benchmark_score": avg_benchmark,
        "avg_greedy_score": avg_greedy,
        "n_passing_greedy": n_passing_greedy,
        "n_passing_avg": n_passing_avg,
        "n_episodes": ep_label,
        "tasks": use_tasks,
    }
    results_path = Path(plot_dir) / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved benchmark: {results_path}")

    # ---- Plot --------------------------------------------------------------
    _plot_training_curves(history, Path(plot_dir) / "training_curves.png")

    return results


def _plot_training_curves(history: dict, out_path: Path) -> None:
    """Plot episode score and rolling average. Falls back gracefully if no display."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np

        episodes = history["episode"]
        scores = history["score"]

        # Rolling average with window 20
        window = 20
        rolling = [
            sum(scores[max(0, i - window): i + 1]) / min(i + 1, window)
            for i in range(len(scores))
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "InvoiceTriageEnv — REINFORCE Training (PyTorch)",
            fontsize=14,
            fontweight="bold",
        )

        # Score curve
        ax = axes[0]
        ax.plot(episodes, scores, alpha=0.3, color="#4A90E2", linewidth=0.8, label="Episode score")
        ax.plot(episodes, rolling, color="#E2574A", linewidth=2.0, label=f"Rolling avg (n={window})")
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Target (0.80)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Normalised Score (0–1)")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Steps per episode
        ax2 = axes[1]
        steps_rolling = [
            sum(history["steps"][max(0, i - window): i + 1]) / min(i + 1, window)
            for i in range(len(history["steps"]))
        ]
        ax2.plot(episodes, history["steps"], alpha=0.2, color="#9B59B6", linewidth=0.8)
        ax2.plot(episodes, steps_rolling, color="#9B59B6", linewidth=2.0, label=f"Rolling avg (n={window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps per Episode")
        ax2.set_title("Efficiency (fewer steps = better)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved training curve: {out_path}")

    except ImportError:
        print("  ⚠ matplotlib not installed — skipping plot. pip install matplotlib")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch REINFORCE agent on InvoiceTriageEnv"
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task IDs to train on (default: all). Use 'all' for all tasks.",
    )
    parser.add_argument(
        "--save-dir",
        default="outputs/checkpoints",
        help="Directory to save policy checkpoints",
    )
    parser.add_argument(
        "--plot-dir", default="outputs", help="Directory to save training plots"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pt file to warm-start from (loads weights only)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip training — load checkpoint and run benchmark evaluation only.",
    )
    args = parser.parse_args()

    task_ids = None
    if args.tasks and args.tasks != ["all"]:
        unknown = [t for t in args.tasks if t not in ALL_TASKS]
        if unknown:
            print(f"Unknown task IDs: {unknown}")
            print(f"Available: {list(ALL_TASKS.keys())}")
            return
        task_ids = args.tasks

    t0 = time.time()
    results = train(
        n_episodes=0 if args.benchmark_only else args.episodes,
        task_ids=task_ids,
        save_dir=args.save_dir,
        plot_dir=args.plot_dir,
        seed=args.seed,
        resume_checkpoint=args.resume,
    )
    elapsed = time.time() - t0
    print(f"  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
