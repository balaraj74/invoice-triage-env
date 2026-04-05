#!/usr/bin/env bash
# ============================================================================
# run.sh — One-command launcher for InvoiceTriageEnv
# ============================================================================
#
# Usage:
#   ./run.sh              # Run inference (LLM agent evaluation)
#   ./run.sh server       # Start the API server (localhost:7860)
#   ./run.sh test         # Run all unit tests
#   ./run.sh validate     # Run OpenEnv validator
#   ./run.sh dashboard    # Serve the dashboard locally
#   ./run.sh all          # Run tests + validate + inference
#
# Environment variables (set in .env or export):
#   GEMINI_API_KEY        — Google Gemini API key
#   API_BASE_URL          — Override API endpoint (default: Gemini OpenAI-compat)
#   MODEL_NAME            — Override model name (default: gemini-2.5-flash)
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---- Load .env ----
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# ---- Activate venv ----
if [[ -d .venv ]]; then
    source .venv/bin/activate
elif command -v uv &>/dev/null; then
    echo "Creating virtual environment with uv..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
fi

# ---- Defaults: Vertex AI (uses GCP billing credits, no free-tier limits) ----
GCP_PROJECT="${GOOGLE_CLOUD_PROJECT:-agrisence-1dc30}"
GCP_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
export API_BASE_URL="${API_BASE_URL:-https://${GCP_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/${GCP_PROJECT}/locations/${GCP_LOCATION}/endpoints/openapi/}"
export MODEL_NAME="${MODEL_NAME:-google/gemini-2.5-flash}"

# Use gcloud access token for Vertex AI auth (bills to GCP credits)
if [[ "$API_BASE_URL" == *"aiplatform.googleapis.com"* ]]; then
    export HF_TOKEN="$(gcloud auth print-access-token 2>/dev/null)"
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: gcloud auth failed. Run: gcloud auth login"
        exit 1
    fi
else
    export HF_TOKEN="${GEMINI_API_KEY:-${HF_TOKEN:-}}"
fi

# ---- Print config ----
print_config() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  InvoiceTriageEnv                                       ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  API:   ${API_BASE_URL:0:50}"
    echo "║  Model: ${MODEL_NAME}"
    echo "║  Key:   ${HF_TOKEN:0:10}..."
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

# ---- Commands ----
cmd_inference() {
    print_config
    echo "🚀 Running LLM agent inference on all 6 tasks..."
    echo ""
    python3 inference.py
}

cmd_server() {
    echo "🖥️  Starting API server on http://localhost:7860"
    python3 -m invoice_triage_env.server.app
}

cmd_test() {
    echo "🧪 Running unit tests..."
    pytest tests/ -v --tb=short
}

cmd_validate() {
    echo "✅ Running OpenEnv validator..."
    openenv validate
}

cmd_dashboard() {
    echo "📊 Serving dashboard at http://localhost:8080"
    python3 -m invoice_triage_env.dashboard.serve
}

cmd_all() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1/3: Unit Tests"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cmd_test
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2/3: OpenEnv Validate"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cmd_validate
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 3/3: LLM Inference"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cmd_inference
}

# ---- Dispatch ----
case "${1:-inference}" in
    inference|run)  cmd_inference ;;
    server)         cmd_server ;;
    test|tests)     cmd_test ;;
    validate)       cmd_validate ;;
    dashboard)      cmd_dashboard ;;
    all)            cmd_all ;;
    -h|--help|help)
        echo "Usage: ./run.sh [inference|server|test|validate|dashboard|all]"
        echo ""
        echo "Commands:"
        echo "  inference   Run LLM agent on all 6 tasks (default)"
        echo "  server      Start the API server"
        echo "  test        Run unit tests"
        echo "  validate    Run OpenEnv validator"
        echo "  dashboard   Serve performance dashboard"
        echo "  all         Run tests + validate + inference"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run ./run.sh --help for usage"
        exit 1
        ;;
esac
