FROM python:3.11-slim

# 1. Install System Dependencies (Required for Graphviz and ChromaDB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
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
# Volumes in docker-compose.yml mount knowledge_base/ and chroma_db/
# from the EC2 host, so those are NOT baked into the image.
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]