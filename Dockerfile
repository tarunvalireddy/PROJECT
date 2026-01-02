FROM python:3.11-slim

WORKDIR /APP

COPY requirements.txt .

RUN pip install --no-cache-dir  -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "gunicorn","CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST.wsgi:application","--bind","0.0.0.0:8000" ]
