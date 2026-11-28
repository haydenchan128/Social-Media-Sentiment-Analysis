FROM python:3.14-slim

WORKDIR /app

# Install dashboard dependencies
COPY requirements-dashboard.txt .

RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy project files
COPY . .

# Streamlit's default port
EXPOSE 8501

# Launch the dashboard
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]