# Crypto Price Prediction Using LSTM & XGBoost

This is a Django-based web application for predicting cryptocurrency prices using Machine Learning models such as **LSTM** and **XGBoost**.  
The application can be run either **locally** or **inside a Docker container using Gunicorn**.

---

## Requirements

### For Local Development
- Python 3.10+
- pip
- virtualenv

### For Container Execution
- Docker

---

## Local Development (Without Docker)

### 1. Clone the project
```bash
git clone <repository-url>
cd PROJECT


2. Create and activate virtual environment

Linux / macOS


python3 -m venv venv
source venv/bin/activate


Windows

python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Apply migrations
python manage.py migrate

5. Run the application
python manage.py runserver 0.0.0.0:8000

6. Access the app
http://127.0.0.1:8000


Docker Container Execution
1. Build the Docker image
docker build -t crypto-price-predictor .

2. Run the container
docker run -d -p 8000:8000 crypto-price-predictor

3. Access the app
http://localhost:8000


or (on EC2)

http://<EC2_PUBLIC_IP>:8000

Application Server

The application runs using Gunicorn inside the container:

gunicorn CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST.wsgi:application \
--bind 0.0.0.0:8000

Important Notes

Do not use Django runserver in production

Configure ALLOWED_HOSTS correctly in settings.py

Use Nginx in front of Gunicorn for ports 80/443

Store secrets using environment variables

Useful Docker Commands

View running containers:

docker ps


View logs:

docker logs <container_id>


Stop container:

docker stop <container_id>

Tech Stack

Django

TensorFlow

XGBoost

Scikit-learn

Gunicorn

Docker

Python 3.11




