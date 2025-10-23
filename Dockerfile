FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY /app .
RUN ls
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
