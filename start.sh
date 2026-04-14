#!/bin/bash

# Navigate to the correct base directory
cd "$(dirname "$0")"

# Load environment variables from .env safely
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -o allexport
  source .env
  set +o allexport
else
  echo ".env file not found! Starting without it."
fi

# Cleanup function to terminate background processes on Ctrl+C
cleanup() {
  echo ""
  echo "Stopping backend and frontend servers..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit
}

# Catch the termination signals
trap cleanup SIGINT SIGTERM

echo "Starting FastAPI backend (Port 8000)..."
source .venv/bin/activate
uvicorn invoice_triage_env.server.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting Next.js frontend (Port 3000)..."
cd dashboard-nextjs
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are successfully running!"
echo "➡️  Frontend: http://localhost:3000"
echo "➡️  Backend:  http://localhost:8000"
echo "⏸️  Press Ctrl+C to stop both servers."
echo ""

# Keep the script running to catch Ctrl+C
wait $BACKEND_PID $FRONTEND_PID
