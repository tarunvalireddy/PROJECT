
```
# Crypto Price Predictor

This project uses LSTM and XGBoost to predict cryptocurrency prices. Follow the instructions below to set up the environment and run the application.

## Local Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd PROJECT

```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate

```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Database & Execution

```bash
# Apply migrations
python manage.py migrate

# Run the application
python manage.py runserver 0.0.0.0:8000

```

**Access the app at:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Docker Container Execution

### 1. Build and Run

```bash
# Build the Docker image
docker build -t crypto-price-predictor .

# Run the container
docker run -d -p 8000:8000 crypto-price-predictor

```

### 2. Accessing the Application

* **Local:** [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000)
* **EC2:** `http://<EC2_PUBLIC_IP>:8000`

### 3. Application Server

The application runs using **Gunicorn** inside the container:

```bash
gunicorn CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST.wsgi:application \
--bind 0.0.0.0:8000

```

---

## Tech Stack

* **Framework:** Django
* **ML Libraries:** TensorFlow, XGBoost, Scikit-learn
* **Server/Deployment:** Gunicorn, Docker
* **Language:** Python 3.11

---

## Useful Docker Commands

* **View running containers:** `docker ps`
* **View logs:** `docker logs <container_id>`
* **Stop container:** `docker stop <container_id>`

---

## Important Notes

* **Production:** Do **not** use Django `runserver` in production.
* **Security:** Configure `ALLOWED_HOSTS` correctly in `settings.py`.
* **Proxy:** Use **Nginx** in front of Gunicorn for ports 80/443.
* **Environment:** Store secrets using environment variables.

```

```
