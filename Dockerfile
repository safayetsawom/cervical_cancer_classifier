FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
   EXPOSE 8080
   CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
