FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn tensorflow pillow numpy python-multipart pydantic joblib scikit-learn

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
