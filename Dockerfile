FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for caching
COPY invoice_triage_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the full project
COPY invoice_triage_env/ /app/invoice_triage_env/
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py
COPY pyproject.toml /app/pyproject.toml
COPY outputs/ /app/outputs/

# Install the package itself
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "invoice_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
