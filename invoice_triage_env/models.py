"""Typed Pydantic models for the InvoiceTriageEnv OpenEnv environment.

Defines Action, Observation, and State used across server and client.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import (
    Action,
    Observation,
    State,
)


# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------


class InvoiceCategory(str, Enum):
    """High-level spend category for an invoice."""

    SUPPLIES = "supplies"
    TRAVEL = "travel"
    SOFTWARE = "software"
    CONSULTING = "consulting"
    UTILITIES = "utilities"
    MAINTENANCE = "maintenance"
    MARKETING = "marketing"
    EQUIPMENT = "equipment"
    OTHER = "other"


class Priority(str, Enum):
    """Processing priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ActionType(str, Enum):
    """Available agent actions."""

    CATEGORIZE = "categorize"
    SET_PRIORITY = "set_priority"
    FLAG_ISSUE = "flag_issue"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    EXTRACT_FIELD = "extract_field"
    VALIDATE_MATCH = "validate_match"
    SUBMIT_DECISION = "submit_decision"


class IssueType(str, Enum):
    """Types of issues an agent can flag."""

    AMOUNT_MISMATCH = "amount_mismatch"
    MISSING_PO = "missing_po"
    DUPLICATE_INVOICE = "duplicate_invoice"
    VENDOR_MISMATCH = "vendor_mismatch"
    DATE_ANOMALY = "date_anomaly"
    TAX_ERROR = "tax_error"
    MISSING_APPROVAL = "missing_approval"
    OVER_BUDGET = "over_budget"
    SUSPICIOUS_VENDOR = "suspicious_vendor"


# ---------------------------------------------------------------------------
# Line Item
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(description="Item description")
    quantity: float = Field(description="Quantity ordered")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Line total (quantity × unit_price)")
    po_line_ref: Optional[str] = Field(
        default=None, description="Reference to PO line item"
    )


# ---------------------------------------------------------------------------
# Invoice Document
# ---------------------------------------------------------------------------


class InvoiceDocument(BaseModel):
    """A single invoice document presented to the agent."""

    invoice_id: str = Field(description="Unique invoice identifier")
    vendor_name: str = Field(description="Vendor/supplier name")
    vendor_id: str = Field(description="Vendor account identifier")
    invoice_date: str = Field(description="Invoice date (YYYY-MM-DD)")
    due_date: str = Field(description="Payment due date (YYYY-MM-DD)")
    currency: str = Field(default="USD", description="Invoice currency")
    subtotal: float = Field(description="Sum of line items before tax")
    tax_amount: float = Field(description="Tax amount")
    total_amount: float = Field(description="Total invoice amount")
    po_number: Optional[str] = Field(
        default=None, description="Linked purchase order number"
    )
    line_items: List[LineItem] = Field(
        default_factory=list, description="Individual line items"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or context"
    )


# ---------------------------------------------------------------------------
# Purchase Order (for validation)
# ---------------------------------------------------------------------------


class PurchaseOrder(BaseModel):
    """A purchase order to validate invoices against."""

    po_number: str = Field(description="Purchase order number")
    vendor_id: str = Field(description="Expected vendor ID")
    vendor_name: str = Field(description="Expected vendor name")
    total_amount: float = Field(description="Approved total amount")
    items: List[LineItem] = Field(
        default_factory=list, description="Ordered items"
    )
    approved_by: str = Field(description="Approver name")
    budget_code: str = Field(description="Budget/cost center code")
    remaining_budget: float = Field(description="Remaining budget for this code")


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------


class InvoiceAction(Action):
    """Action the agent takes to process the current invoice.

    The agent selects an action_type and provides the relevant payload.
    """

    action_type: ActionType = Field(description="Type of action to perform")

    # Payloads — only relevant fields need to be set per action_type
    category: Optional[InvoiceCategory] = Field(
        default=None, description="Category for CATEGORIZE action"
    )
    priority: Optional[Priority] = Field(
        default=None, description="Priority for SET_PRIORITY action"
    )
    issue_type: Optional[IssueType] = Field(
        default=None, description="Issue type for FLAG_ISSUE action"
    )
    issue_description: Optional[str] = Field(
        default=None, description="Free-text description of the flagged issue"
    )
    field_name: Optional[str] = Field(
        default=None, description="Field name for EXTRACT_FIELD action"
    )
    field_value: Optional[str] = Field(
        default=None, description="Extracted value for EXTRACT_FIELD action"
    )
    match_result: Optional[bool] = Field(
        default=None,
        description="Whether invoice matches PO for VALIDATE_MATCH action",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for APPROVE / REJECT / ESCALATE actions",
    )


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------


class InvoiceObservation(Observation):
    """Observation returned to the agent after each step.

    Contains the current invoice, available PO data, task goal,
    processing history, and feedback from the last action.
    """

    goal: str = Field(
        default="", description="Natural-language description of the current task"
    )
    invoice: Optional[InvoiceDocument] = Field(
        default=None, description="The invoice to process"
    )
    purchase_order: Optional[PurchaseOrder] = Field(
        default=None, description="Related purchase order (if available)"
    )
    historical_invoices: List[InvoiceDocument] = Field(
        default_factory=list,
        description="Past invoices from the same vendor (for duplicate detection)",
    )
    available_actions: List[str] = Field(
        default_factory=list, description="Actions the agent can take right now"
    )
    last_action_feedback: str = Field(
        default="", description="Feedback from the last action taken"
    )
    last_action_error: Optional[str] = Field(
        default=None, description="Error message if last action failed"
    )
    progress: Dict[str, bool] = Field(
        default_factory=dict,
        description="Checklist of completed sub-tasks in the episode",
    )
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed")


# ---------------------------------------------------------------------------
# State Model
# ---------------------------------------------------------------------------


class InvoiceState(State):
    """Internal environment state for tracking episode progress."""

    task_id: str = Field(default="", description="Current task identifier")
    task_difficulty: str = Field(
        default="easy", description="Difficulty level of current task"
    )
    completed_subtasks: List[str] = Field(
        default_factory=list, description="Subtasks completed so far"
    )
    required_subtasks: List[str] = Field(
        default_factory=list, description="Subtasks required for full credit"
    )
    issues_found: List[str] = Field(
        default_factory=list, description="Issues flagged by agent"
    )
    issues_expected: List[str] = Field(
        default_factory=list, description="Issues that exist in the invoice"
    )
    final_decision: Optional[str] = Field(
        default=None, description="Final decision (approve/reject/escalate)"
    )
    expected_decision: Optional[str] = Field(
        default=None, description="Expected correct decision"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Sum of rewards earned so far"
    )
