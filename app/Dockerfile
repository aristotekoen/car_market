FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt  && \
    pip install --no-cache-dir --force-reinstall catboost numpy pandas scikit-learn

EXPOSE 8080

CMD ["streamlit", "run", "predict.py", "--server.port", "8080", "--server.address", "0.0.0.0"]