FROM python:3.11-slim

# 1. Install System Dependencies (Required for Graphviz & ChromeDB)
RUN apt-get update && apt-get install -y \
    graphviz \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up App
WORKDIR /app
COPY requirements.txt .

# 3. Install Python Libs
RUN pip install --no-cache-dir -r requirements.txt


# 4. Copy Code
COPY . .
# 4b. Copy Streamlit secrets
COPY .streamlit /app/.streamlit

# 5. Run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]