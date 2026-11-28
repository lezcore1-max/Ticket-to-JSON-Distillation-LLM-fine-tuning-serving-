FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY labeling/ labeling/
COPY training/__init__.py training/__init__.py
COPY training/dataset.py training/dataset.py
COPY serving/__init__.py serving/__init__.py
COPY serving/app.py serving/app.py

EXPOSE 8080

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]

