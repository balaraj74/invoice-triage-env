# InvoiceTriageEnv 🧾

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for training AI agents to process, validate, and adjudicate supplier invoices in an accounts-payable workflow.

## Why Invoice Triage?

Invoice processing costs enterprises **$15–$40 per invoice** manually. Agents trained in this environment learn the full AP triage loop:

1. **Categorize** spend (supplies, consulting, software, …)
2. **Set priority** (low → urgent)
3. **Extract** key fields (vendor, amount, PO#)
4. **Validate** against purchase orders
5. **Flag issues** — amount mismatches, duplicates, tax errors, suspicious vendors, budget blowouts
6. **Decide** — approve, reject, or escalate

## Quick Start

### Local (no server)

```bash
# Install
pip install openenv-core

# Run the demo agent on all 6 tasks
PYTHONPATH=. python -m invoice_triage_env.examples.run_agent
```

### With HTTP Server

```bash
# Start the server
PYTHONPATH=. uvicorn invoice_triage_env.server.app:app --host 0.0.0.0 --port 8000

# Use the typed client
python -c "
from invoice_triage_env.client import InvoiceTriageClient
client = InvoiceTriageClient('http://localhost:8000')
obs = client.reset()
print(obs.goal)
"
```

### Docker

```bash
cd invoice_triage_env
docker build -t invoice-triage-env -f server/Dockerfile .
docker run -p 8000:8000 invoice-triage-env
```

## Tasks

| ID | Difficulty | Issues | Expected |
|----|-----------|--------|----------|
| `easy_approve_clean` | Easy | None | Approve |
| `easy_reject_no_po` | Easy | Missing PO | Reject |
| `medium_amount_mismatch` | Medium | Rate overcharge | Escalate |
| `medium_duplicate_detection` | Medium | Duplicate invoice | Reject |
| `hard_multi_issue_fraud` | Hard | 6 concurrent issues | Reject |
| `hard_suspicious_vendor` | Hard | Budget + escalating costs | Escalate |

## Action Space

| Action | Required Fields | Reward |
|--------|----------------|--------|
| `categorize` | `category` | +1.0 / −0.5 |
| `set_priority` | `priority` | +0.5 / −0.3 |
| `extract_field` | `field_name`, `field_value` | +0.5 / −0.2 |
| `validate_match` | `match_result` (bool) | +1.0 / −0.5 |
| `flag_issue` | `issue_type` | +1.5 / −1.0 |
| `approve` | `reason` | +3.0 / −2.0 |
| `reject` | `reason` | +3.0 / −2.0 |
| `escalate` | `reason` | +3.0 / −2.0 |

Per-step cost: **−0.05** (encourages efficiency).

## Project Structure

```
invoice_triage_env/
├── __init__.py
├── models.py              # Pydantic Action/Observation/State
├── tasks.py               # 6 richly-detailed task scenarios
├── client.py              # Typed HTTP client
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI entrypoint
│   ├── invoice_triage_environment.py  # Core Environment
│   ├── Dockerfile
│   └── requirements.txt
└── examples/
    └── run_agent.py       # Rule-based demo agent
```

## License

BSD-3-Clause
