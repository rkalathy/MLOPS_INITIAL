FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.pkl app.py ./
EXPOSE 5000 8001
CMD ["python", "app.py"]
