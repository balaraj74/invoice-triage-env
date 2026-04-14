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


# =============================================================================
# EASY TASKS — Additional
# =============================================================================

TASK_EASY_UTILITIES = TaskDefinition(
    task_id="easy_utilities_approve",
    difficulty="easy",
    goal=(
        "Process this monthly utilities invoice from CityPower Electric. "
        "PO is on file. Categorize, set priority, validate, and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-CP-2026-0311",
        vendor_name="CityPower Electric",
        vendor_id="VENDOR-0008",
        invoice_date="2026-03-01",
        due_date="2026-03-31",
        currency="USD",
        subtotal=4200.00,
        tax_amount=336.00,
        total_amount=4536.00,
        po_number="PO-2026-0100",
        line_items=[
            LineItem(description="Electricity Building A March 2026", quantity=1, unit_price=2800.00, total=2800.00, po_line_ref="PO-2026-0100-L1"),
            LineItem(description="Electricity Building B March 2026", quantity=1, unit_price=1400.00, total=1400.00, po_line_ref="PO-2026-0100-L2"),
        ],
        notes="Monthly utility billing.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-0100",
        vendor_id="VENDOR-0008",
        vendor_name="CityPower Electric",
        total_amount=4500.00,
        items=[LineItem(description="Monthly electricity all buildings", quantity=1, unit_price=4500.00, total=4500.00)],
        approved_by="Facilities Manager",
        budget_code="FAC-UTIL-2026",
        remaining_budget=50000.00,
    ),
    historical_invoices=[],
    expected_category="utilities",
    expected_priority="low",
    expected_issues=[],
    expected_decision="approve",
    required_subtasks=["categorized", "priority_set", "po_validated", "decision_made"],
    max_steps=10,
)

TASK_EASY_TRAVEL = TaskDefinition(
    task_id="easy_travel_approve",
    difficulty="easy",
    goal=(
        "Process this travel expense invoice from BusinessTravel Corp "
        "for the Q1 sales conference. Categorize, set priority, and approve."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-BT-44021",
        vendor_name="BusinessTravel Corp",
        vendor_id="VENDOR-0055",
        invoice_date="2026-03-18",
        due_date="2026-04-18",
        currency="USD",
        subtotal=7800.00,
        tax_amount=0.00,
        total_amount=7800.00,
        po_number="PO-2026-TR-001",
        line_items=[
            LineItem(description="Flights x4 staff return", quantity=4, unit_price=850.00, total=3400.00),
            LineItem(description="Hotel 4 nights x4 staff", quantity=16, unit_price=195.00, total=3120.00),
            LineItem(description="Ground transport", quantity=1, unit_price=1280.00, total=1280.00),
        ],
        notes="Q1 Sales Leadership Conference Chicago.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-TR-001",
        vendor_id="VENDOR-0055",
        vendor_name="BusinessTravel Corp",
        total_amount=8000.00,
        items=[LineItem(description="Q1 sales conference travel", quantity=1, unit_price=8000.00, total=8000.00)],
        approved_by="VP Sales",
        budget_code="SALES-TRAVEL-2026",
        remaining_budget=22000.00,
    ),
    historical_invoices=[],
    expected_category="travel",
    expected_priority="low",
    expected_issues=[],
    expected_decision="approve",
    required_subtasks=["categorized", "priority_set", "po_validated", "decision_made"],
    max_steps=10,
)

TASK_EASY_MISSING_APPROVAL = TaskDefinition(
    task_id="easy_missing_approval",
    difficulty="easy",
    goal=(
        "Process this invoice from SafeGuard Security Services. "
        "The purchase order has NOT been approved by anyone. "
        "Flag the missing approval issue and reject."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-SG-8810",
        vendor_name="SafeGuard Security Services",
        vendor_id="VENDOR-0201",
        invoice_date="2026-03-20",
        due_date="2026-04-20",
        currency="USD",
        subtotal=9500.00,
        tax_amount=760.00,
        total_amount=10260.00,
        po_number="PO-2026-SEC-002",
        line_items=[
            LineItem(description="On-site security personnel March", quantity=1, unit_price=9500.00, total=9500.00),
        ],
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-SEC-002",
        vendor_id="VENDOR-0201",
        vendor_name="SafeGuard Security Services",
        total_amount=9500.00,
        items=[LineItem(description="Security services", quantity=1, unit_price=9500.00, total=9500.00)],
        approved_by="",
        budget_code="FAC-SEC-2026",
        remaining_budget=30000.00,
    ),
    historical_invoices=[],
    expected_category="maintenance",
    expected_priority="medium",
    expected_issues=["missing_approval"],
    expected_decision="reject",
    required_subtasks=["categorized", "priority_set", "issue_flagged", "decision_made"],
    max_steps=12,
)


# =============================================================================
# MEDIUM TASKS — Additional
# =============================================================================

TASK_MEDIUM_FOREX = TaskDefinition(
    task_id="medium_fx_currency_risk",
    difficulty="medium",
    goal=(
        "Process this software invoice from GlobalSoft GmbH Germany. "
        "Invoice is in EUR but the purchase order is in USD. "
        "Check for currency mismatch and amount mismatch, then decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-GS-DE-2026-0041",
        vendor_name="GlobalSoft GmbH",
        vendor_id="VENDOR-0300",
        invoice_date="2026-03-12",
        due_date="2026-04-12",
        currency="EUR",
        subtotal=22000.00,
        tax_amount=4180.00,
        total_amount=26180.00,
        po_number="PO-2026-IT-200",
        line_items=[
            LineItem(description="Enterprise SaaS Q1 license EUR", quantity=1, unit_price=22000.00, total=22000.00),
        ],
        notes="Invoice in EUR. Exchange rate not specified.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-IT-200",
        vendor_id="VENDOR-0300",
        vendor_name="GlobalSoft GmbH",
        total_amount=24000.00,
        items=[LineItem(description="Enterprise SaaS Q1 license", quantity=1, unit_price=24000.00, total=24000.00)],
        approved_by="CTO",
        budget_code="IT-SW-INTL-2026",
        remaining_budget=30000.00,
    ),
    historical_invoices=[],
    expected_category="software",
    expected_priority="high",
    expected_issues=["amount_mismatch"],
    expected_decision="escalate",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=15,
)

TASK_MEDIUM_EQUIPMENT = TaskDefinition(
    task_id="medium_equipment_over_budget",
    difficulty="medium",
    goal=(
        "Process this invoice from TechGear Pro for IT equipment. "
        "The remaining budget is critically low at 2000. "
        "Validate amounts, flag the over-budget issue, and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-TGP-2026-122",
        vendor_name="TechGear Pro",
        vendor_id="VENDOR-0180",
        invoice_date="2026-03-14",
        due_date="2026-04-14",
        currency="USD",
        subtotal=38500.00,
        tax_amount=3080.00,
        total_amount=41580.00,
        po_number="PO-2026-IT-150",
        line_items=[
            LineItem(description="MacBook Pro 16 x10 units", quantity=10, unit_price=2800.00, total=28000.00),
            LineItem(description="External monitors x15", quantity=15, unit_price=550.00, total=8250.00),
            LineItem(description="Docking stations x10", quantity=10, unit_price=225.00, total=2250.00),
        ],
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-IT-150",
        vendor_id="VENDOR-0180",
        vendor_name="TechGear Pro",
        total_amount=40000.00,
        items=[LineItem(description="IT equipment refresh", quantity=1, unit_price=40000.00, total=40000.00)],
        approved_by="IT Director",
        budget_code="IT-EQUIP-2026",
        remaining_budget=2000.00,
    ),
    historical_invoices=[],
    expected_category="equipment",
    expected_priority="high",
    expected_issues=["over_budget"],
    expected_decision="escalate",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=15,
)

TASK_MEDIUM_SUBSCRIPTION_OVERBILL = TaskDefinition(
    task_id="medium_subscription_overbill",
    difficulty="medium",
    goal=(
        "Process this software invoice from CloudStack Inc. "
        "Historical invoices show a monthly rate of 4500 for 12 months "
        "but this month they are billing 6800. Investigate the mismatch."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-CS-2026-Q1-03",
        vendor_name="CloudStack Inc",
        vendor_id="VENDOR-0088",
        invoice_date="2026-03-01",
        due_date="2026-03-31",
        currency="USD",
        subtotal=6800.00,
        tax_amount=544.00,
        total_amount=7344.00,
        po_number="PO-2025-SAAS-010",
        line_items=[
            LineItem(description="Platform subscription March 2026", quantity=1, unit_price=4500.00, total=4500.00),
            LineItem(description="Enhanced support package", quantity=1, unit_price=2300.00, total=2300.00),
        ],
        notes="Enhanced support added per verbal agreement.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2025-SAAS-010",
        vendor_id="VENDOR-0088",
        vendor_name="CloudStack Inc",
        total_amount=54000.00,
        items=[LineItem(description="Annual SaaS subscription", quantity=12, unit_price=4500.00, total=54000.00)],
        approved_by="VP Engineering",
        budget_code="IT-SAAS-2026",
        remaining_budget=9000.00,
    ),
    historical_invoices=[
        InvoiceDocument(
            invoice_id="INV-CS-2026-Q1-01", vendor_name="CloudStack Inc", vendor_id="VENDOR-0088",
            invoice_date="2026-01-01", due_date="2026-01-31", currency="USD",
            subtotal=4500.00, tax_amount=360.00, total_amount=4860.00, po_number="PO-2025-SAAS-010",
            line_items=[LineItem(description="Platform subscription Jan", quantity=1, unit_price=4500.00, total=4500.00)],
        ),
        InvoiceDocument(
            invoice_id="INV-CS-2026-Q1-02", vendor_name="CloudStack Inc", vendor_id="VENDOR-0088",
            invoice_date="2026-02-01", due_date="2026-02-28", currency="USD",
            subtotal=4500.00, tax_amount=360.00, total_amount=4860.00, po_number="PO-2025-SAAS-010",
            line_items=[LineItem(description="Platform subscription Feb", quantity=1, unit_price=4500.00, total=4500.00)],
        ),
    ],
    expected_category="software",
    expected_priority="high",
    expected_issues=["amount_mismatch"],
    expected_decision="escalate",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=15,
)

TASK_MEDIUM_SPLIT_INVOICE = TaskDefinition(
    task_id="medium_split_invoice_abuse",
    difficulty="medium",
    goal=(
        "Process this invoice from FastFix Contractors. "
        "Three invoices each for 9800 arrived from this vendor this week "
        "with no purchase order. This looks like split invoicing to avoid "
        "the 10000 approval limit. Flag suspicious vendor and escalate."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-FF-2026-0081",
        vendor_name="FastFix Contractors",
        vendor_id="VENDOR-0505",
        invoice_date="2026-03-22",
        due_date="2026-04-22",
        currency="USD",
        subtotal=9800.00,
        tax_amount=784.00,
        total_amount=10584.00,
        po_number=None,
        line_items=[
            LineItem(description="Office renovation Phase 1", quantity=1, unit_price=9800.00, total=9800.00),
        ],
        notes="Invoice 1 of 3 for office renovation project.",
    ),
    purchase_order=None,
    historical_invoices=[
        InvoiceDocument(
            invoice_id="INV-FF-2026-0079", vendor_name="FastFix Contractors", vendor_id="VENDOR-0505",
            invoice_date="2026-03-19", due_date="2026-04-19", currency="USD",
            subtotal=9800.00, tax_amount=784.00, total_amount=10584.00, po_number=None,
            line_items=[LineItem(description="Office renovation prep work", quantity=1, unit_price=9800.00, total=9800.00)],
        ),
        InvoiceDocument(
            invoice_id="INV-FF-2026-0080", vendor_name="FastFix Contractors", vendor_id="VENDOR-0505",
            invoice_date="2026-03-20", due_date="2026-04-20", currency="USD",
            subtotal=9800.00, tax_amount=784.00, total_amount=10584.00, po_number=None,
            line_items=[LineItem(description="Office renovation materials", quantity=1, unit_price=9800.00, total=9800.00)],
        ),
    ],
    expected_category="maintenance",
    expected_priority="urgent",
    expected_issues=["missing_po", "suspicious_vendor"],
    expected_decision="escalate",
    required_subtasks=["categorized", "priority_set", "issue_flagged", "decision_made"],
    max_steps=15,
)


# =============================================================================
# HARD TASKS — Additional
# =============================================================================

TASK_HARD_GHOST_VENDOR = TaskDefinition(
    task_id="hard_ghost_vendor",
    difficulty="hard",
    goal=(
        "Process this invoice from Phantom Analytics LLC. "
        "This vendor has no history with the company. "
        "The PO has no approver and is over budget. "
        "Payment note references a personal account. "
        "Flag all suspicious issues and reject."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-PA-0001",
        vendor_name="Phantom Analytics LLC",
        vendor_id="VND-UNKNOWN-001",
        invoice_date="2026-03-28",
        due_date="2026-04-04",
        currency="USD",
        subtotal=75000.00,
        tax_amount=6000.00,
        total_amount=81000.00,
        po_number="PO-2026-PHNT-01",
        line_items=[
            LineItem(description="Strategic consulting services", quantity=1, unit_price=75000.00, total=75000.00),
        ],
        notes="Urgent. Bank wire only. Reference JD-PERSONAL-ACCT.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-PHNT-01",
        vendor_id="VND-UNKNOWN-001",
        vendor_name="Phantom Analytics LLC",
        total_amount=75000.00,
        items=[LineItem(description="Consulting", quantity=1, unit_price=75000.00, total=75000.00)],
        approved_by="",
        budget_code="CEO-DISC-2026",
        remaining_budget=1000.00,
    ),
    historical_invoices=[],
    expected_category="consulting",
    expected_priority="urgent",
    expected_issues=["suspicious_vendor", "missing_approval", "over_budget", "date_anomaly"],
    expected_decision="reject",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=20,
)

TASK_HARD_TAX_FRAUD = TaskDefinition(
    task_id="hard_tax_fraud_pattern",
    difficulty="hard",
    goal=(
        "Process this invoice from MegaSupply Corp. "
        "The invoice subtotal is 0 but line items total 12000. "
        "Tax is 1800 calculated on incorrect base. "
        "Validate all arithmetic, flag tax error and amount mismatch, and reject."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-MS-2026-7721",
        vendor_name="MegaSupply Corp",
        vendor_id="VENDOR-0601",
        invoice_date="2026-03-10",
        due_date="2026-04-10",
        currency="USD",
        subtotal=0.00,
        tax_amount=1800.00,
        total_amount=13800.00,
        po_number="PO-2026-SUP-099",
        line_items=[
            LineItem(description="Industrial cleaning supplies", quantity=100, unit_price=60.00, total=6000.00),
            LineItem(description="Safety equipment", quantity=30, unit_price=200.00, total=6000.00),
        ],
        notes="Urgent reorder. Price locked.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-SUP-099",
        vendor_id="VENDOR-0601",
        vendor_name="MegaSupply Corp",
        total_amount=12000.00,
        items=[LineItem(description="Cleaning and safety supplies", quantity=1, unit_price=12000.00, total=12000.00)],
        approved_by="Operations Director",
        budget_code="OPS-SUPPLIES-2026",
        remaining_budget=15000.00,
    ),
    historical_invoices=[],
    expected_category="supplies",
    expected_priority="urgent",
    expected_issues=["tax_error", "amount_mismatch"],
    expected_decision="reject",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=18,
)

TASK_HARD_HEALTHCARE = TaskDefinition(
    task_id="hard_healthcare_complex",
    difficulty="hard",
    goal=(
        "Process this invoice from MedDevice Partners for medical equipment. "
        "Multiple line item rates differ from the purchase order. "
        "One item was already billed in January so it is a duplicate. "
        "Amount would exceed department budget. Flag all issues and decide."
    ),
    invoice=InvoiceDocument(
        invoice_id="INV-MDP-2026-0441",
        vendor_name="MedDevice Partners",
        vendor_id="VENDOR-0350",
        invoice_date="2026-03-15",
        due_date="2026-04-15",
        currency="USD",
        subtotal=128000.00,
        tax_amount=0.00,
        total_amount=128000.00,
        po_number="PO-2026-MED-010",
        line_items=[
            LineItem(description="Ultrasound unit Model X-500", quantity=2, unit_price=45000.00, total=90000.00, po_line_ref="PO-2026-MED-010-L1"),
            LineItem(description="Patient monitoring systems", quantity=4, unit_price=7500.00, total=30000.00, po_line_ref="PO-2026-MED-010-L2"),
            LineItem(description="Installation and calibration", quantity=1, unit_price=8000.00, total=8000.00, po_line_ref="PO-2026-MED-010-L3"),
        ],
        notes="Healthcare equipment FDA certified.",
    ),
    purchase_order=PurchaseOrder(
        po_number="PO-2026-MED-010",
        vendor_id="VENDOR-0350",
        vendor_name="MedDevice Partners",
        total_amount=110000.00,
        items=[
            LineItem(description="Ultrasound unit Model X-500", quantity=2, unit_price=40000.00, total=80000.00),
            LineItem(description="Patient monitoring systems", quantity=4, unit_price=7000.00, total=28000.00),
            LineItem(description="Installation and calibration", quantity=1, unit_price=2000.00, total=2000.00),
        ],
        approved_by="Chief Medical Officer",
        budget_code="MED-EQUIP-2026",
        remaining_budget=15000.00,
    ),
    historical_invoices=[
        InvoiceDocument(
            invoice_id="INV-MDP-2026-0399", vendor_name="MedDevice Partners", vendor_id="VENDOR-0350",
            invoice_date="2026-01-20", due_date="2026-02-20", currency="USD",
            subtotal=8000.00, tax_amount=0.00, total_amount=8000.00, po_number="PO-2026-MED-005",
            line_items=[LineItem(description="Installation and calibration", quantity=1, unit_price=8000.00, total=8000.00)],
        ),
    ],
    expected_category="equipment",
    expected_priority="urgent",
    expected_issues=["amount_mismatch", "over_budget", "duplicate_invoice"],
    expected_decision="escalate",
    required_subtasks=["categorized", "priority_set", "po_validated", "issue_flagged", "decision_made"],
    max_steps=20,
)


# =============================================================================
# Extended Registry — 16 total tasks (5 easy, 6 medium, 5 hard)
# =============================================================================

# Replace the previous ALL_TASKS and TASKS_BY_DIFFICULTY with extended versions
ALL_TASKS = {
    t.task_id: t
    for t in [
        TASK_EASY_APPROVE,
        TASK_EASY_REJECT,
        TASK_EASY_UTILITIES,
        TASK_EASY_TRAVEL,
        TASK_EASY_MISSING_APPROVAL,
        TASK_MEDIUM_MISMATCH,
        TASK_MEDIUM_DUPLICATE,
        TASK_MEDIUM_FOREX,
        TASK_MEDIUM_EQUIPMENT,
        TASK_MEDIUM_SUBSCRIPTION_OVERBILL,
        TASK_MEDIUM_SPLIT_INVOICE,
        TASK_HARD_MULTI_ISSUE,
        TASK_HARD_BUDGET_SUSPICIOUS,
        TASK_HARD_GHOST_VENDOR,
        TASK_HARD_TAX_FRAUD,
        TASK_HARD_HEALTHCARE,
    ]
}

TASKS_BY_DIFFICULTY = {
    "easy": [TASK_EASY_APPROVE, TASK_EASY_REJECT, TASK_EASY_UTILITIES,
             TASK_EASY_TRAVEL, TASK_EASY_MISSING_APPROVAL],
    "medium": [TASK_MEDIUM_MISMATCH, TASK_MEDIUM_DUPLICATE, TASK_MEDIUM_FOREX,
               TASK_MEDIUM_EQUIPMENT, TASK_MEDIUM_SUBSCRIPTION_OVERBILL,
               TASK_MEDIUM_SPLIT_INVOICE],
    "hard": [TASK_HARD_MULTI_ISSUE, TASK_HARD_BUDGET_SUSPICIOUS, TASK_HARD_GHOST_VENDOR,
             TASK_HARD_TAX_FRAUD, TASK_HARD_HEALTHCARE],
}
