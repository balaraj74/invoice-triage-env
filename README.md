---
title: InvoiceTriageEnv
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - invoice-triage
  - ai-agent
  - accounts-payable
license: bsd-3-clause
---

<div align="center">

# 🧾 InvoiceTriageEnv

**A production-grade OpenEnv environment for training AI agents to process, validate, and adjudicate supplier invoices.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/balarajr/invoice-triage-env)

[Live Dashboard](https://balarajr-invoice-triage-env.hf.space) · [API Docs](#api-reference) · [Quick Start](#-quick-start) · [Architecture](#-architecture)

</div>

---

## 📌 Overview

Invoice processing costs enterprises **$15–$40 per invoice** manually. InvoiceTriageEnv simulates the complete accounts-payable triage workflow, challenging AI agents to:

1. **Categorize** spend (supplies, consulting, software, marketing, …)
2. **Prioritize** urgency (low → urgent)
3. **Extract** key fields (vendor name, amount, PO number, tax)
4. **Validate** against purchase orders — detect amount mismatches
5. **Flag issues** — duplicates, tax errors, suspicious vendors, budget violations
6. **Decide** — approve ✅, reject ❌, or escalate ⚠️

This is not a toy environment. Every task is modeled after real enterprise AP workflows with realistic invoices, purchase orders, historical data, and multi-issue fraud scenarios.

---

## 🏆 Baseline Results

| Model | Avg Score | Accuracy | Avg Steps |
|-------|-----------|----------|-----------|
| **Gemini 2.5 Flash (Vertex AI)** | **1.0000** | **6/6 (100%)** | **8.8** |
| Rule-based heuristic | 0.47 | 4/6 (67%) | 5.2 |

<details>
<summary><strong>Per-task breakdown</strong></summary>

| Task ID | Difficulty | Score | Steps | Decision |
|---------|-----------|-------|-------|----------|
| `easy_approve_clean` | 🟢 Easy | 1.0000 | 7 | ✅ approve |
| `easy_reject_no_po` | 🟢 Easy | 1.0000 | 6 | ❌ reject |
| `medium_amount_mismatch` | 🟡 Medium | 1.0000 | 8 | ⚠️ escalate |
| `medium_duplicate_detection` | 🟡 Medium | 1.0000 | 8 | ❌ reject |
| `hard_multi_issue_fraud` | 🔴 Hard | 1.0000 | 14 | ❌ reject |
| `hard_suspicious_vendor` | 🔴 Hard | 1.0000 | 10 | ⚠️ escalate |

</details>

---

## 🚀 Quick Start

### Option 1: One-Command Script (Recommended)

```bash
git clone https://huggingface.co/spaces/balarajr/invoice-triage-env
cd invoice-triage-env

# Install
pip install -e ".[server,dev]"

# Run the LLM agent evaluation
./run.sh
```

### Option 2: Manual Setup

```bash
# 1. Clone & install
git clone https://huggingface.co/spaces/balarajr/invoice-triage-env
cd invoice-triage-env
pip install -e ".[server,dev]"

# 2. Configure (choose one):

# --- Vertex AI (recommended, uses GCP billing credits) ---
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
# (run.sh handles gcloud auth automatically)

# --- OpenAI ---
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."

# --- Gemini (free tier, rate-limited) ---
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.5-flash"
export HF_TOKEN="AIzaSy..."

# 3. Run inference
python inference.py
```

### Option 3: Docker

```bash
docker build -t invoice-triage-env .
docker run -p 8000:8000 invoice-triage-env

# API available at http://localhost:8000
```

---

## 🎮 `run.sh` — All-in-One Launcher

```bash
./run.sh              # Run LLM agent inference on all 6 tasks (default)
./run.sh server       # Start the API server on localhost:8000
./run.sh test         # Run all 38 unit tests
./run.sh validate     # Run OpenEnv spec validator
./run.sh dashboard    # Serve the performance dashboard locally
./run.sh all          # Run tests → validate → inference (full pipeline)
```

---

## 🧠 How the Agent Works

The agent interacts with the environment through a **step loop**:

```
┌─────────┐    reset()     ┌─────────────┐
│  Agent   │ ──────────────►│ Environment │
│  (LLM)   │◄──────────────│             │
│          │  observation   │  Invoice +  │
│          │    + goal      │  PO + Hist  │
│          │                └──────┬──────┘
│          │    step(action)       │
│          │ ─────────────────────►│
│          │◄─────────────────────│
│          │  observation +       │
│          │  reward + feedback   │
│          │                      │
│          │  (repeat until done) │
└─────────┘                       │
```

Each episode:
1. Agent receives an invoice, purchase order (if any), and historical invoices
2. Agent takes actions: categorize → prioritize → extract → validate → flag → decide
3. Environment returns feedback + reward after each action
4. Episode ends when agent makes a final decision (approve/reject/escalate)
5. Score is normalized to **0.0–1.0** based on correctness

---

## 📋 Tasks

### Easy Tasks

| Task | Scenario | Expected Decision |
|------|----------|-------------------|
| `easy_approve_clean` | Clean invoice from Acme Office Supplies with matching PO | **Approve** |
| `easy_reject_no_po` | Invoice from QuickPrint Services with no purchase order | **Reject** |

### Medium Tasks

| Task | Scenario | Expected Decision |
|------|----------|-------------------|
| `medium_amount_mismatch` | TechWave Consulting invoice with rate overcharge vs PO | **Escalate** |
| `medium_duplicate_detection` | GreenLeaf Maintenance invoice that duplicates a past payment | **Reject** |

### Hard Tasks

| Task | Scenario | Expected Decision |
|------|----------|-------------------|
| `hard_multi_issue_fraud` | NovaStar Solutions invoice with **6 concurrent issues**: amount mismatch, duplicate, vendor mismatch, date anomaly, tax error, over-budget | **Reject** |
| `hard_suspicious_vendor` | Apex Digital Marketing with escalating costs, amount mismatch, budget violation, and suspicious vendor pattern | **Escalate** |

---

## 🎯 Action Space

| Action | Required Fields | Reward (Correct) | Reward (Wrong) |
|--------|----------------|-------------------|----------------|
| `categorize` | `category` | +1.0 | −0.5 |
| `set_priority` | `priority` | +0.5 | −0.3 |
| `extract_field` | `field_name`, `field_value` | +0.5 | −0.2 |
| `validate_match` | `match_result` (bool) | +1.0 | −0.5 |
| `flag_issue` | `issue_type`, `issue_description` | +1.5 | −1.0 |
| `approve` | `reason` | +3.0 | −2.0 |
| `reject` | `reason` | +3.0 | −2.0 |
| `escalate` | `reason` | +3.0 | −2.0 |

**Per-step cost:** −0.05 (encourages efficiency, excluded from grading score)

### Categories
`supplies` · `travel` · `software` · `consulting` · `utilities` · `maintenance` · `marketing` · `equipment` · `other`

### Priorities
`low` · `medium` · `high` · `urgent`

### Issue Types
`amount_mismatch` · `missing_po` · `duplicate_invoice` · `vendor_mismatch` · `date_anomaly` · `tax_error` · `missing_approval` · `over_budget` · `suspicious_vendor`

---

## 👁️ Observation Space

Each observation returned by `reset()` and `step()` contains:

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Normalized score (0.0–1.0) |
| `goal` | `str` | Natural-language task description |
| `invoice` | `Invoice` | Full invoice with line items, dates, amounts |
| `purchase_order` | `PurchaseOrder?` | PO document for validation (null if none) |
| `historical_invoices` | `Invoice[]` | Past invoices from same vendor |
| `available_actions` | `str[]` | Legal actions at current step |
| `progress` | `dict` | Subtask completion checklist |
| `last_action_feedback` | `str` | Human-readable feedback from last action |
| `last_action_error` | `str?` | Error message if last action was invalid |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum allowed steps (20) |

---

## 📊 Reward Function

The reward system provides **partial progress signals** — agents are rewarded incrementally for each correct action, not just at the end.

### Scoring Formula

```
normalized_score = correctness_reward / max_possible_reward
```

Where `correctness_reward` excludes step costs (step costs are a training signal for efficiency, not a correctness penalty).

### Reward Breakdown

| Component | Value | When Awarded |
|-----------|-------|--------------|
| Correct category | +1.0 | `categorize` with right category |
| Correct priority | +0.5 | `set_priority` with right priority |
| Correct extraction | +0.5 | `extract_field` with matching value |
| PO match correct | +1.0 | `validate_match` with correct bool |
| Issue flagged correctly | +1.5 | `flag_issue` with expected issue |
| Correct decision | +3.0 | approve/reject/escalate correctly |
| All subtasks complete | +1.0 | Bonus when all required steps done |
| Missed issue penalty | −0.8 | Per missed issue at decision time |
| False positive flag | −1.0 | Flagging an issue that doesn't exist |
| Wrong decision | −2.0 | Incorrect approve/reject/escalate |
| Step cost | −0.05 | Per step (training signal only) |

---

## 🔌 API Reference

### `POST /reset`

Reset the environment to a new episode.

```json
// Request
{ "seed": 42, "task_id": "easy_approve_clean" }

// Response
{
  "observation": {
    "done": false,
    "reward": 0.0,
    "goal": "Process this invoice from Acme Office Supplies...",
    "invoice": { ... },
    "purchase_order": { ... },
    "available_actions": ["categorize", "set_priority", ...],
    "progress": { "categorized": false, "priority_set": false, ... }
  }
}
```

### `POST /step`

Take an action in the environment.

```json
// Request
{
  "action": {
    "action_type": "categorize",
    "category": "supplies"
  }
}

// Response
{
  "observation": {
    "done": false,
    "reward": 0.1538,
    "last_action_feedback": "Correct! Category set to 'supplies'.",
    "available_actions": ["set_priority", "extract_field", ...],
    "progress": { "categorized": true, "priority_set": false, ... }
  }
}
```

### `GET /state`

Get the current environment state.

```json
{
  "task_id": "easy_approve_clean",
  "step_count": 3,
  "done": false,
  "cumulative_reward": 2.35
}
```

### Python Client

```python
from invoice_triage_env.client import InvoiceTriageClient
from invoice_triage_env.models import InvoiceAction, ActionType

client = InvoiceTriageClient("http://localhost:8000")

# Reset
obs = client.reset(seed=42, task_id="easy_approve_clean")
print(obs.goal)

# Step
obs = client.step(InvoiceAction(
    action_type=ActionType.CATEGORIZE,
    category="supplies"
))
print(f"Reward: {obs.reward}, Feedback: {obs.last_action_feedback}")
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    InvoiceTriageEnv                       │
│                                                          │
│   ┌──────────────┐    ┌────────────────────────────┐    │
│   │  FastAPI      │    │  InvoiceTriageEnvironment  │    │
│   │  Server       │───►│                            │    │
│   │  (app.py)     │    │  • reset() / step()        │    │
│   └──────────────┘    │  • Reward computation       │    │
│                        │  • Score normalization      │    │
│   ┌──────────────┐    │  • Subtask tracking         │    │
│   │  Dashboard    │    └──────────┬─────────────────┘    │
│   │  (index.html) │              │                       │
│   └──────────────┘    ┌──────────▼─────────────────┐    │
│                        │  Tasks (tasks.py)           │    │
│   ┌──────────────┐    │                             │    │
│   │  Typed Client │    │  6 scenarios with invoices, │    │
│   │  (client.py)  │    │  POs, and ground truth      │    │
│   └──────────────┘    └─────────────────────────────┘    │
│                                                          │
│   ┌──────────────┐    ┌─────────────────────────────┐   │
│   │  Models       │    │  Inference Script            │   │
│   │  (models.py)  │    │  (inference.py)              │   │
│   │  Pydantic v2  │    │  OpenAI-compatible client    │   │
│   └──────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── run.sh                        # 🚀 One-command launcher (inference/server/test/validate)
├── inference.py                  # 🤖 Baseline inference script (OpenAI-compatible)
├── Dockerfile                    # 🐳 Production container (HF Spaces)
├── openenv.yaml                  # 📋 OpenEnv manifest
├── pyproject.toml                # 📦 Package configuration
├── uv.lock                       # 🔒 Dependency lock file
├── .env                          # 🔑 Environment variables (gitignored)
│
├── invoice_triage_env/           # 📂 Core Python package
│   ├── __init__.py
│   ├── models.py                 #    Pydantic Action/Observation/State models
│   ├── tasks.py                  #    6 richly-detailed task scenarios
│   ├── client.py                 #    Typed HTTP client
│   ├── evaluate.py               #    Evaluation harness
│   │
│   ├── server/                   # 🖥️ API server
│   │   ├── app.py                #    FastAPI entrypoint
│   │   ├── invoice_triage_environment.py  # Core environment logic
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── examples/                 # 💡 Example agents
│   │   ├── run_agent.py          #    Rule-based demo agent
│   │   └── run_llm_agent.py      #    LLM-powered agent
│   │
│   └── dashboard/                # 📊 Performance dashboard
│       ├── __init__.py
│       ├── index.html            #    Live performance visualization
│       └── serve.py              #    Local dashboard server
│
├── server/                       # 🔗 Root server wrapper (openenv validate)
│   ├── __init__.py
│   └── app.py
│
└── tests/                        # ✅ Unit tests (38 passing)
    ├── __init__.py
    └── test_environment.py
```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project for Vertex AI | `agrisence-1dc30` |
| `GOOGLE_CLOUD_LOCATION` | GCP region | `us-central1` |
| `API_BASE_URL` | LLM API endpoint | Vertex AI endpoint |
| `MODEL_NAME` | Model identifier | `google/gemini-2.5-flash` |
| `HF_TOKEN` | API key (auto-set by `run.sh` for Vertex AI) | — |
| `GEMINI_API_KEY` | Google AI Studio key (free tier fallback) | — |

---

## ✅ OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| Full `step()` / `reset()` / `state()` API | ✅ |
| Typed Pydantic models (Action, Observation, State) | ✅ |
| `openenv.yaml` manifest | ✅ |
| Minimum 3 tasks with graders | ✅ (6 tasks) |
| Scores normalized to 0.0–1.0 | ✅ |
| Meaningful reward with partial progress | ✅ |
| Baseline inference script with reproducible scores | ✅ |
| Working Dockerfile for HF Spaces | ✅ |
| `openenv validate` passes | ✅ |
| Easy → Medium → Hard difficulty progression | ✅ |

---

## 🧪 Testing

```bash
# Run all tests
./run.sh test

# Or directly
pytest tests/ -v --tb=short

# Expected output: 38 passed
```

### Test Coverage

- Environment reset and state management
- All 8 action types with correct/incorrect inputs
- Reward computation and normalization (0.0–1.0)
- Subtask tracking and completion bonuses
- Edge cases: invalid actions, duplicate flags, step limits
- All 6 task scenarios end-to-end

---

## 🐳 Deployment

### Hugging Face Spaces (Live)

The environment is deployed at [balarajr/invoice-triage-env](https://huggingface.co/spaces/balarajr/invoice-triage-env) with:
- Docker SDK
- Auto-rebuilds on push
- Live performance dashboard at root URL
- API endpoints at `/reset`, `/step`, `/state`

### Self-Hosted

```bash
docker build -t invoice-triage-env .
docker run -p 8000:8000 -e HF_TOKEN=your-key invoice-triage-env
```

---

## 🔧 Development

```bash
# Install in dev mode
pip install -e ".[server,dev]"

# Run tests
pytest tests/ -v

# Lint + type check
ruff check .
mypy invoice_triage_env/

# Validate OpenEnv spec
openenv validate

# Full pipeline
./run.sh all
```

---

## 📈 Building Your Own Agent

Want to beat the baseline? Here's how to get started:

```python
from invoice_triage_env.server.invoice_triage_environment import InvoiceTriageEnvironment
from invoice_triage_env.models import InvoiceAction, ActionType

env = InvoiceTriageEnvironment(task_id="easy_approve_clean")
obs = env.reset(seed=42)

# Your agent logic here
while not obs.done:
    action = your_agent.decide(obs)  # Return an InvoiceAction
    obs = env.step(action)

print(f"Final score: {obs.reward}")  # 0.0–1.0
```

### Tips for High Scores

1. **Always categorize + set priority first** — they're required subtasks
2. **Extract fields before validating** — gives the agent context
3. **Check PO match carefully** — `match_result=False` when amounts differ
4. **Flag ALL issues** — missed issues get −0.8 penalty each
5. **Don't flag false positives** — each costs −1.0
6. **Use historical invoices** — essential for duplicate detection

---

## 📜 License

BSD-3-Clause

---

<div align="center">

**Built for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv)** · Made with ❤️ by [@balarajr](https://huggingface.co/balarajr)

</div>
