FROM python:3.11-slim

# 1. Install System Dependencies (Required for ChromaDB native build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up App
WORKDIR /app
COPY requirements.txt .

# 3. Install Python Libs
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# 4. Copy Code
# The agent/ and tools/ subdirectories are included automatically.
# chroma_db/ is excluded via .dockerignore (persisted on EC2 via volume mount).
COPY . .

# 5. Non-root user (UID 1000 matches the ubuntu user on EC2 so the
#    bind-mounted chroma_db/ volume remains writable).
RUN addgroup --system --gid 1000 appuser \
 && adduser --system --uid 1000 --ingroup appuser appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]