FROM python:3.11-slim AS build

WORKDIR /APP

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN  pip install --upgrade pip \
     &&  pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM gcr.io/distroless/python3

WORKDIR /APP

COPY --from=build /install /usr/local

COPY . .

EXPOSE 8000

CMD [ "/usr/local/bin/gunicorn","CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST.wsgi:application","--bind","0.0.0.0:8000" ]
    
