"""Task definitions for the InvoiceTriageEnv.

Each task is a scenario with:
- An invoice (possibly with embedded issues)
- An optional purchase order
- Historical invoices (for duplicate detection)
- A ground-truth expected outcome
- A difficulty rating (easy / medium / hard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from invoice_triage_env.models import (
    InvoiceDocument,
    LineItem,
    PurchaseOrder,
)


@dataclass
class TaskDefinition:
    """A single evaluation task."""

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    goal: str
    invoice: InvoiceDocument
    purchase_order: Optional[PurchaseOrder]
    historical_invoices: List[InvoiceDocument]
    expected_category: str
    expected_priority: str
    expected_issues: List[str]
    expected_decision: str  # "approve" | "reject" | "escalate"
    expected_extractions: Dict[str, str] = field(default_factory=dict)
    required_subtasks: List[str] = field(default_factory=list)
    max_steps: int = 15


# =============================================================================
# EASY TASKS — Straightforward invoices, obvious decisions
# =============================================================================


TASK_EASY_APPROVE = TaskDefinition(
    task_id="easy_approve_clean",
    difficulty="easy",
    goal=(
        "Process this invoice from Acme Office Supplies. "
        "Categorize it, set priority, validate against the purchase order, "
        "and make a final decision (approve, reject, or escalate)."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-2026-0042",
        vendor_name="Acme Office Supplies",
        vendor_id="VENDOR-0012",
        invoice_date="2026-03-15",
        due_date="2026-04-15",
        currency="USD",
        subtotal=450.00,
        tax_amount=36.00,
        total_amount=486.00,
        po_number="PO-2026-1001",
        line_items=[
            LineItem(
                description="A4 Paper (500 sheets × 10 reams)",
                quantity=10,
                unit_price=12.00,
                total=120.00,
                po_line_ref="PO-2026-1001-L1",
            ),
            LineItem(
                description="Ballpoint Pens (box of 50)",
                quantity=5,
                unit_price=18.00,
                total=90.00,
                po_line_ref="PO-2026-1001-L2",
            ),
            LineItem(
                description="Whiteboard Markers (pack of 12)",
                quantity=10,
                unit_price=24.00,
                total=240.00,
                po_line_ref="PO-2026-1001-L3",
            ),
        ],
        notes="Standard monthly office supply order.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-1001",
        vendor_id="VENDOR-0012",
        vendor_name="Acme Office Supplies",
        total_amount=500.00,
        items=[
            LineItem(
                description="A4 Paper (500 sheets × 10 reams)",
                quantity=10,
                unit_price=12.00,
                total=120.00,
            ),
            LineItem(
                description="Ballpoint Pens (box of 50)",
                quantity=5,
                unit_price=18.00,
                total=90.00,
            ),
            LineItem(
                description="Whiteboard Markers (pack of 12)",
                quantity=10,
                unit_price=24.00,
                total=240.00,
            ),
        ],
        approved_by="Sarah Johnson",
        budget_code="DEPT-OPS-2026",
        remaining_budget=12500.00,
    ),
    historical_invoices=[],
    expected_category="supplies",
    expected_priority="low",
    expected_issues=[],
    expected_decision="approve",
    expected_extractions={
        "vendor_name": "Acme Office Supplies",
        "total_amount": "486.00",
        "po_number": "PO-2026-1001",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "po_validated",
        "decision_made",
    ],
    max_steps=12,
)


TASK_EASY_REJECT = TaskDefinition(
    task_id="easy_reject_no_po",
    difficulty="easy",
    goal=(
        "Process this invoice from QuickPrint Services. "
        "No purchase order exists for this invoice. "
        "Categorize, set priority, and decide what to do."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-QP-8831",
        vendor_name="QuickPrint Services",
        vendor_id="VENDOR-0099",
        invoice_date="2026-03-20",
        due_date="2026-04-20",
        currency="USD",
        subtotal=3200.00,
        tax_amount=256.00,
        total_amount=3456.00,
        po_number=None,
        line_items=[
            LineItem(
                description="Large format poster printing (A1)",
                quantity=100,
                unit_price=32.00,
                total=3200.00,
            ),
        ],
        notes="Urgent marketing campaign materials.",
    ),
    purchase_order=None,
    historical_invoices=[],
    expected_category="marketing",
    expected_priority="medium",
    expected_issues=["missing_po"],
    expected_decision="reject",
    expected_extractions={
        "vendor_name": "QuickPrint Services",
        "total_amount": "3456.00",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "issue_flagged",
        "decision_made",
    ],
    max_steps=12,
)


# =============================================================================
# MEDIUM TASKS — Subtle discrepancies requiring validation
# =============================================================================


TASK_MEDIUM_MISMATCH = TaskDefinition(
    task_id="medium_amount_mismatch",
    difficulty="medium",
    goal=(
        "Process this invoice from TechWave Consulting. "
        "A purchase order is attached. Carefully validate all amounts, "
        "check for discrepancies, categorize, prioritise, and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-TW-2026-0155",
        vendor_name="TechWave Consulting",
        vendor_id="VENDOR-0045",
        invoice_date="2026-03-10",
        due_date="2026-04-10",
        currency="USD",
        subtotal=15750.00,
        tax_amount=1260.00,
        total_amount=17010.00,
        po_number="PO-2026-2050",
        line_items=[
            LineItem(
                description="Cloud architecture review (senior consultant, 40h)",
                quantity=40,
                unit_price=225.00,
                total=9000.00,
                po_line_ref="PO-2026-2050-L1",
            ),
            LineItem(
                description="Security audit (mid-level, 30h)",
                quantity=30,
                unit_price=175.00,
                total=5250.00,
                po_line_ref="PO-2026-2050-L2",
            ),
            LineItem(
                description="Documentation (junior, 10h)",
                quantity=10,
                unit_price=150.00,
                total=1500.00,
                po_line_ref="PO-2026-2050-L3",
            ),
        ],
        notes="Phase 2 of cloud migration project.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-2050",
        vendor_id="VENDOR-0045",
        vendor_name="TechWave Consulting",
        total_amount=14500.00,  # ← PO total is less than invoice subtotal
        items=[
            LineItem(
                description="Cloud architecture review (senior consultant, 40h)",
                quantity=40,
                unit_price=200.00,  # ← Rate is $200, not $225
                total=8000.00,
            ),
            LineItem(
                description="Security audit (mid-level, 30h)",
                quantity=30,
                unit_price=175.00,
                total=5250.00,
            ),
            LineItem(
                description="Documentation (junior, 10h)",
                quantity=10,
                unit_price=125.00,  # ← Rate is $125, not $150
                total=1250.00,
            ),
        ],
        approved_by="David Chen",
        budget_code="PROJ-CLOUD-2026",
        remaining_budget=25000.00,
    ),
    historical_invoices=[],
    expected_category="consulting",
    expected_priority="high",
    expected_issues=["amount_mismatch"],
    expected_decision="escalate",
    expected_extractions={
        "vendor_name": "TechWave Consulting",
        "total_amount": "17010.00",
        "po_number": "PO-2026-2050",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "po_validated",
        "issue_flagged",
        "decision_made",
    ],
    max_steps=15,
)


TASK_MEDIUM_DUPLICATE = TaskDefinition(
    task_id="medium_duplicate_detection",
    difficulty="medium",
    goal=(
        "Process this invoice from GreenLeaf Maintenance. "
        "Historical invoices from the same vendor are provided. "
        "Check for duplicates, validate, categorize, and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-GL-4420",
        vendor_name="GreenLeaf Maintenance",
        vendor_id="VENDOR-0033",
        invoice_date="2026-03-22",
        due_date="2026-04-22",
        currency="USD",
        subtotal=2800.00,
        tax_amount=224.00,
        total_amount=3024.00,
        po_number="PO-2026-3010",
        line_items=[
            LineItem(
                description="Monthly grounds maintenance — March 2026",
                quantity=1,
                unit_price=2000.00,
                total=2000.00,
            ),
            LineItem(
                description="Emergency tree removal (storm damage)",
                quantity=1,
                unit_price=800.00,
                total=800.00,
            ),
        ],
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-3010",
        vendor_id="VENDOR-0033",
        vendor_name="GreenLeaf Maintenance",
        total_amount=3200.00,
        items=[
            LineItem(
                description="Monthly grounds maintenance — March 2026",
                quantity=1,
                unit_price=2000.00,
                total=2000.00,
            ),
            LineItem(
                description="Emergency tree removal (storm damage)",
                quantity=1,
                unit_price=800.00,
                total=800.00,
            ),
        ],
        approved_by="Lisa Park",
        budget_code="FAC-MAINT-2026",
        remaining_budget=18000.00,
    ),
    historical_invoices=[
        # This is a DUPLICATE — same items, same amounts, a week earlier
        InvoiceDocument(
            invoice_id="INV-GL-4418",
            vendor_name="GreenLeaf Maintenance",
            vendor_id="VENDOR-0033",
            invoice_date="2026-03-15",
            due_date="2026-04-15",
            currency="USD",
            subtotal=2800.00,
            tax_amount=224.00,
            total_amount=3024.00,
            po_number="PO-2026-3010",
            line_items=[
                LineItem(
                    description="Monthly grounds maintenance — March 2026",
                    quantity=1,
                    unit_price=2000.00,
                    total=2000.00,
                ),
                LineItem(
                    description="Emergency tree removal (storm damage)",
                    quantity=1,
                    unit_price=800.00,
                    total=800.00,
                ),
            ],
        ),
        InvoiceDocument(
            invoice_id="INV-GL-4310",
            vendor_name="GreenLeaf Maintenance",
            vendor_id="VENDOR-0033",
            invoice_date="2026-02-15",
            due_date="2026-03-15",
            currency="USD",
            subtotal=2000.00,
            tax_amount=160.00,
            total_amount=2160.00,
            po_number="PO-2026-3005",
            line_items=[
                LineItem(
                    description="Monthly grounds maintenance — February 2026",
                    quantity=1,
                    unit_price=2000.00,
                    total=2000.00,
                ),
            ],
        ),
    ],
    expected_category="maintenance",
    expected_priority="high",
    expected_issues=["duplicate_invoice"],
    expected_decision="reject",
    expected_extractions={
        "vendor_name": "GreenLeaf Maintenance",
        "total_amount": "3024.00",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "po_validated",
        "issue_flagged",
        "decision_made",
    ],
    max_steps=15,
)


# =============================================================================
# HARD TASKS — Multiple issues, tax errors, suspicious patterns
# =============================================================================


TASK_HARD_MULTI_ISSUE = TaskDefinition(
    task_id="hard_multi_issue_fraud",
    difficulty="hard",
    goal=(
        "Process this invoice from NovaStar Solutions. "
        "A matching PO is available. Historical invoices are provided. "
        "This invoice may contain MULTIPLE issues. "
        "Thoroughly validate every field, check line item math, "
        "verify vendor identity, detect any duplicates, flag ALL issues, "
        "categorize, prioritise, and submit your final decision with reasoning."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-NS-7742",
        vendor_name="NovaStar Solutions",  # Note slight name difference from PO
        vendor_id="VENDOR-0078",
        invoice_date="2026-03-25",
        due_date="2026-03-27",  # ← Only 2 days to pay — suspicious
        currency="USD",
        subtotal=28500.00,
        tax_amount=2565.00,  # ← 9% tax rate — should be 8% ($2280)
        total_amount=31065.00,
        po_number="PO-2026-5500",
        line_items=[
            LineItem(
                description="Enterprise license — AI Analytics Platform (annual)",
                quantity=1,
                unit_price=18000.00,
                total=18000.00,
                po_line_ref="PO-2026-5500-L1",
            ),
            LineItem(
                description="Implementation & onboarding (40h)",
                quantity=40,
                unit_price=200.00,
                total=8000.00,
                po_line_ref="PO-2026-5500-L2",
            ),
            LineItem(
                description="Priority support add-on (12 months)",
                quantity=1,
                unit_price=2500.00,
                total=2500.00,
                po_line_ref="PO-2026-5500-L3",
            ),
        ],
        notes="Please process immediately — license expires if not renewed by March 28.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-5500",
        vendor_id="VENDOR-0078",
        vendor_name="Nova Star Solutions Inc.",  # ← Name doesn't match exactly
        total_amount=26000.00,  # ← PO total is less than invoice subtotal
        items=[
            LineItem(
                description="Enterprise license — AI Analytics Platform (annual)",
                quantity=1,
                unit_price=18000.00,
                total=18000.00,
            ),
            LineItem(
                description="Implementation & onboarding (40h)",
                quantity=40,
                unit_price=200.00,
                total=8000.00,
            ),
            # ← No priority support add-on in original PO
        ],
        approved_by="Maria Santos",
        budget_code="IT-SW-2026",
        remaining_budget=8000.00,  # ← Invoice would blow the budget
    ),
    historical_invoices=[
        InvoiceDocument(
            invoice_id="INV-NS-7700",
            vendor_name="NovaStar Solutions",
            vendor_id="VENDOR-0078",
            invoice_date="2026-03-20",
            due_date="2026-04-20",
            currency="USD",
            subtotal=18000.00,
            tax_amount=1440.00,
            total_amount=19440.00,
            po_number="PO-2026-5500",
            line_items=[
                LineItem(
                    description="Enterprise license — AI Analytics Platform (annual)",
                    quantity=1,
                    unit_price=18000.00,
                    total=18000.00,
                ),
            ],
            notes="Initial license payment.",
        ),
    ],
    expected_category="software",
    expected_priority="urgent",
    expected_issues=[
        "amount_mismatch",
        "tax_error",
        "vendor_mismatch",
        "date_anomaly",
        "over_budget",
        "duplicate_invoice",
    ],
    expected_decision="reject",
    expected_extractions={
        "vendor_name": "NovaStar Solutions",
        "total_amount": "31065.00",
        "po_number": "PO-2026-5500",
        "tax_amount": "2565.00",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "po_validated",
        "issue_flagged",
        "decision_made",
    ],
    max_steps=20,
)


TASK_HARD_BUDGET_SUSPICIOUS = TaskDefinition(
    task_id="hard_suspicious_vendor",
    difficulty="hard",
    goal=(
        "Process this invoice from Apex Digital Marketing. "
        "A PO exists but the amounts differ. Historical invoices show "
        "a pattern of escalating costs. Check for budget violations, "
        "suspicious patterns, categorize, prioritise, and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-ADM-2026-089",
        vendor_name="Apex Digital Marketing",
        vendor_id="VENDOR-0112",
        invoice_date="2026-03-24",
        due_date="2026-04-07",
        currency="USD",
        subtotal=45000.00,
        tax_amount=3600.00,
        total_amount=48600.00,
        po_number="PO-2026-6100",
        line_items=[
            LineItem(
                description="SEO campaign management — March 2026",
                quantity=1,
                unit_price=15000.00,
                total=15000.00,
                po_line_ref="PO-2026-6100-L1",
            ),
            LineItem(
                description="PPC ad spend management fee",
                quantity=1,
                unit_price=20000.00,
                total=20000.00,
                po_line_ref="PO-2026-6100-L2",
            ),
            LineItem(
                description="Social media content creation (30 posts)",
                quantity=30,
                unit_price=333.33,
                total=9999.90,
                po_line_ref="PO-2026-6100-L3",
            ),
        ],
        notes="Q1 final invoice. Please note increased scope.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-6100",
        vendor_id="VENDOR-0112",
        vendor_name="Apex Digital Marketing",
        total_amount=30000.00,  # ← PO was for $30k, invoice is $45k
        items=[
            LineItem(
                description="SEO campaign management — March 2026",
                quantity=1,
                unit_price=10000.00,
                total=10000.00,
            ),
            LineItem(
                description="PPC ad spend management fee",
                quantity=1,
                unit_price=15000.00,
                total=15000.00,
            ),
            LineItem(
                description="Social media content creation (20 posts)",
                quantity=20,
                unit_price=250.00,
                total=5000.00,
            ),
        ],
        approved_by="Tom Richards",
        budget_code="MKT-DIGITAL-2026",
        remaining_budget=5000.00,  # ← Way over budget
    ),
    historical_invoices=[
        InvoiceDocument(
            invoice_id="INV-ADM-2026-045",
            vendor_name="Apex Digital Marketing",
            vendor_id="VENDOR-0112",
            invoice_date="2026-01-25",
            due_date="2026-02-25",
            currency="USD",
            subtotal=20000.00,
            tax_amount=1600.00,
            total_amount=21600.00,
            po_number="PO-2026-6050",
            line_items=[
                LineItem(
                    description="SEO + PPC — January 2026",
                    quantity=1,
                    unit_price=20000.00,
                    total=20000.00,
                ),
            ],
        ),
        InvoiceDocument(
            invoice_id="INV-ADM-2026-067",
            vendor_name="Apex Digital Marketing",
            vendor_id="VENDOR-0112",
            invoice_date="2026-02-24",
            due_date="2026-03-24",
            currency="USD",
            subtotal=32000.00,
            tax_amount=2560.00,
            total_amount=34560.00,
            po_number="PO-2026-6075",
            line_items=[
                LineItem(
                    description="SEO + PPC + Social — February 2026",
                    quantity=1,
                    unit_price=32000.00,
                    total=32000.00,
                ),
            ],
        ),
    ],
    expected_category="marketing",
    expected_priority="urgent",
    expected_issues=[
        "amount_mismatch",
        "over_budget",
        "suspicious_vendor",
    ],
    expected_decision="escalate",
    expected_extractions={
        "vendor_name": "Apex Digital Marketing",
        "total_amount": "48600.00",
        "po_number": "PO-2026-6100",
    },
    required_subtasks=[
        "categorized",
        "priority_set",
        "po_validated",
        "issue_flagged",
        "decision_made",
    ],
    max_steps=20,
)


# =============================================================================
# Registry
# =============================================================================


ALL_TASKS: Dict[str, TaskDefinition] = {
    t.task_id: t
    for t in [
        TASK_EASY_APPROVE,
        TASK_EASY_REJECT,
        TASK_MEDIUM_MISMATCH,
        TASK_MEDIUM_DUPLICATE,
        TASK_HARD_MULTI_ISSUE,
        TASK_HARD_BUDGET_SUSPICIOUS,
    ]
}

TASKS_BY_DIFFICULTY: Dict[str, List[TaskDefinition]] = {
    "easy": [TASK_EASY_APPROVE, TASK_EASY_REJECT],
    "medium": [TASK_MEDIUM_MISMATCH, TASK_MEDIUM_DUPLICATE],
    "hard": [TASK_HARD_MULTI_ISSUE, TASK_HARD_BUDGET_SUSPICIOUS],
}
