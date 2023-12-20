FROM python:3.8-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend.py .

COPY count_vectorizer.pkl label_encoder.pkl language_detection_model.pkl /app/

CMD ["python", "backend.py"]
