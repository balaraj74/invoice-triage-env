"""ObservationEncoder — converts InvoiceObservation into a fixed-size torch.Tensor.

Design goals:
  - No external ML dependencies (no sentence-transformers, no BERT).
  - Fully deterministic and fast (< 1 ms per observation).
  - Rich enough to let a policy learn within 500 episodes.

Feature vector layout (total = OBS_DIM = 300):
  [0:9]    one-hot available actions        (9 dims)
  [9:14]   progress checklist              (5 dims, binary)
  [14:16]  step fraction, steps remaining  (2 dims, normalised 0–1)
  [16:28]  invoice numerics                (12 dims)
  [28:36]  purchase-order numerics         (8 dims)
  [36:44]  historical invoice stats        (8 dims)
  [44:300] text hash bag-of-characters     (256 dims)
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, List

# Explicit keywords for rule-based feature extraction from text
_ISSUE_KEYWORDS = [
    "mismatch", "duplicate", "missing po", "no po", "missing approval",
    "suspicious", "tax error", "over budget", "budget", "fraud",
    "ghost", "split", "currency", "forex", "overbill", "abuse",
    "date anomaly", "tax",
]
_CAT_KEYWORDS = [
    "software", "license", "consulting", "marketing", "supplies",
    "maintenance", "equipment", "travel", "utilities",
]
_PRI_KEYWORDS = ["urgent", "high", "medium", "low"]

import torch

if TYPE_CHECKING:
    from invoice_triage_env.models import InvoiceObservation

# Action names — must match ActionType enum order exactly
ACTION_NAMES: List[str] = [
    "categorize",
    "set_priority",
    "flag_issue",
    "approve",
    "reject",
    "escalate",
    "extract_field",
    "validate_match",
    "submit_decision",
]
N_ACTIONS = len(ACTION_NAMES)

# Progress subtask names (consistent ordering)
SUBTASK_KEYS: List[str] = [
    "categorized",
    "priority_set",
    "po_validated",
    "issue_flagged",
    "decision_made",
]
N_SUBTASKS = len(SUBTASK_KEYS)

OBS_DIM = 300 + len(_ISSUE_KEYWORDS) + len(_CAT_KEYWORDS) + len(_PRI_KEYWORDS)  # 300 + 18 + 9 + 4 = 331


def _text_hash_features(text: str, n: int = 256) -> List[float]:
    """Encode arbitrary text as a normalised character-bigram hash vector.

    Cheap alternative to sentence-transformers that requires no extra deps.
    Each bigram is mapped to a bucket in [0, n) via a simple polynomial hash.
    """
    feats = [0.0] * n
    text = text.lower()[:2000]  # truncate for speed
    for i in range(len(text) - 1):
        h = (ord(text[i]) * 31 + ord(text[i + 1])) % n
        feats[h] += 1.0
    total = sum(feats) or 1.0
    return [v / total for v in feats]  # l1-normalise


def _safe_log_amount(amount: float) -> float:
    """Log-scale a dollar amount to ~[0, 1] for amounts up to $10M."""
    return math.log1p(max(0.0, amount)) / math.log1p(10_000_000)


class ObservationEncoder:
    """Stateless encoder: InvoiceObservation → torch.Tensor of shape (OBS_DIM,)."""

    OBS_DIM: int = OBS_DIM

    def encode(self, obs: "InvoiceObservation") -> torch.Tensor:
        """Encode a single observation. Returns shape (OBS_DIM,)."""
        features: List[float] = []

        # ---- [0:9] One-hot available actions --------------------------------
        avail_set = set(obs.available_actions)
        features += [1.0 if a in avail_set else 0.0 for a in ACTION_NAMES]

        # ---- [9:14] Progress checklist --------------------------------------
        features += [
            1.0 if obs.progress.get(k, False) else 0.0 for k in SUBTASK_KEYS
        ]

        # ---- [14:16] Step fraction ------------------------------------------
        max_steps = max(obs.max_steps, 1)
        features.append(obs.step_number / max_steps)            # fraction elapsed
        features.append((max_steps - obs.step_number) / max_steps)  # remaining

        # ---- [16:28] Invoice numerics ----------------------------------------
        inv = obs.invoice
        if inv is not None:
            features += [
                _safe_log_amount(inv.subtotal),
                _safe_log_amount(inv.tax_amount),
                _safe_log_amount(inv.total_amount),
                min(len(inv.line_items) / 10.0, 1.0),       # normalized line count
                1.0 if inv.po_number else 0.0,               # has PO ref
                1.0 if inv.currency == "USD" else 0.0,       # USD flag
                1.0 if inv.currency not in ("USD", "") else 0.0,  # foreign currency
                _safe_log_amount(
                    inv.total_amount - sum(li.total for li in inv.line_items)
                ),  # discrepancy
                min(len(inv.notes or "") / 500.0, 1.0),      # notes length
                1.0 if inv.vendor_id.startswith("V") else 0.0,
                min(len(inv.line_items) / 5.0, 1.0),
                0.0,  # reserved
            ]
        else:
            features += [0.0] * 12

        # ---- [28:44] Purchase-order numerics --------------------------------
        po = obs.purchase_order
        if po is not None:
            inv_total = inv.total_amount if inv else 0.0
            po_total = po.total_amount
            diff = abs(inv_total - po_total)
            features += [
                _safe_log_amount(po_total),
                _safe_log_amount(po.remaining_budget),
                _safe_log_amount(diff),
                min(diff / (po_total + 1e-6), 1.0),          # relative mismatch
                1.0 if diff > 0.01 else 0.0,                 # any mismatch?
                1.0 if po.remaining_budget < inv_total else 0.0,  # over-budget
                min(len(po.items) / 10.0, 1.0),
                0.0,  # reserved
            ]
        else:
            features += [0.0] * 8

        # ---- [36:44] Padding to 44 total so far — historical invoices -------
        hist = obs.historical_invoices
        if hist:
            totals = [h.total_amount for h in hist]
            latest = hist[-1].total_amount
            avg = sum(totals) / len(totals)
            features += [
                min(len(hist) / 5.0, 1.0),                  # how many historical
                _safe_log_amount(avg),                        # avg historical amount
                _safe_log_amount(latest),
                _safe_log_amount(abs(latest - avg)),          # drift from avg
                1.0 if latest > avg * 1.2 else 0.0,          # spike flag
                1.0 if len(hist) >= 3 else 0.0,              # enough history
                min(len(set(h.vendor_id for h in hist)) / 3.0, 1.0),
                0.0,  # reserved
            ]
        else:
            features += [0.0] * 8

        # ---- [60:316] Text hash bag-of-characters ---------------------------
        text_blob = " ".join([
            obs.goal,
            obs.last_action_feedback,
            obs.last_action_error or "",
            inv.vendor_name if inv else "",
            inv.notes or "" if inv else "",
            po.approved_by if po else "",
        ]).lower()
        features += _text_hash_features(text_blob, n=256)

        # ---- Explicit extracted keywords ------------------------------------
        for kw in _ISSUE_KEYWORDS:
            features.append(1.0 if kw in text_blob else 0.0)
        for kw in _CAT_KEYWORDS:
            features.append(1.0 if kw in text_blob else 0.0)
        for kw in _PRI_KEYWORDS:
            features.append(1.0 if kw in text_blob else 0.0)

        assert len(features) == OBS_DIM, f"Expected {OBS_DIM} features, got {len(features)}"
        return torch.tensor(features, dtype=torch.float32)

    def encode_batch(self, obs_list: list) -> torch.Tensor:
        """Encode a batch of observations. Returns shape (batch, OBS_DIM)."""
        return torch.stack([self.encode(o) for o in obs_list])
