from django.shortcuts import render, redirect
from .models import RegisteredUser
from django.core.files.storage import FileSystemStorage

def register_view(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')
        password = request.POST.get('password')
        image = request.FILES.get('image')

        # Basic validation
        if not (name and email and mobile and password and image):
            msg = "All fields are required."
        else:
            # Save image manually
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            img_url = fs.url(filename)

            # Save user with is_active=False
            RegisteredUser.objects.create(
                name=name,
                email=email,
                mobile=mobile,
                password=password,
                image=filename,
                is_active=False
            )
            msg = "Registered successfully! Wait for admin approval."

    return render(request, 'register.html', {'msg': msg})

from django.utils import timezone

from django.utils import timezone
import pytz

def user_login(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            user = RegisteredUser.objects.get(name=name, password=password)
            if user.is_active:
                # Convert current time to IST
                ist = pytz.timezone('Asia/Kolkata')
                local_time = timezone.now().astimezone(ist)

                # Save user info in session
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                request.session['user_image'] = user.image.url  # image URL
                request.session['login_time'] = local_time.strftime('%I:%M:%S %p')

                return redirect('user_homepage')
            else:
                msg = "Your account is not activated yet."
        except RegisteredUser.DoesNotExist:
            msg = "Invalid credentials."

    return render(request, 'user_login.html', {'msg': msg})

def admin_login(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        if name == 'admin' and password == 'admin':
            return redirect('admin_home')
        else:
            msg = "Invalid admin credentials."

    return render(request, 'admin_login.html', {'msg': msg})

def admin_home(request):
    return render(request, 'admin_home.html')
    
def admin_dashboard(request):
    users = RegisteredUser.objects.all()
    return render(request, 'admin_dashboard.html', {'users': users})

def activate_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.is_active = True
    user.save()
    return redirect('admin_dashboard')

def deactivate_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.is_active = False
    user.save()
    return redirect('admin_dashboard')

def delete_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.delete()
    return redirect('admin_dashboard')



def home(request):
    return render(request, 'home.html')

def user_homepage(request):
    if 'user_id' not in request.session:
        # User not logged in, redirect to login page
        return redirect('user_login')

    user_name = request.session.get('user_name')
    user_image = request.session.get('user_image')
    login_time = request.session.get('login_time')

    context = {
        'user_name': user_name,
        'user_image': user_image,
        'login_time': login_time,
    }
    return render(request, 'users/user_homepage.html', context)

def user_logout(request):
    request.session.flush()  # Clears all session data
    return redirect('user_login')



import random
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from .models import RegisteredUser

otp_storage = {}  # Temporary dictionary to store OTPs

def send_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    otp_storage[email] = otp

    subject = "Password Reset OTP"
    message = f"Your OTP for password reset is: {otp}"
    from_email = "saikumardatapoint1@gmail.com"  # Change this to your email
    send_mail(subject, message, from_email, [email])

    return otp

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")

        if RegisteredUser.objects.filter(email=email).exists():
            send_otp(email)
            request.session["reset_email"] = email  # Store email in session
            return redirect("verify_otp")
        else:
            messages.error(request, "Email not registered!")

    return render(request, "forgot_password.html")

def verify_otp(request):
    if request.method == "POST":
        otp_entered = request.POST.get("otp")
        email = request.session.get("reset_email")

        if otp_storage.get(email) and str(otp_storage[email]) == otp_entered:
            return redirect("reset_password")
        else:
            messages.error(request, "Invalid OTP!")

    return render(request, "verify_otp.html")

def reset_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password")
        email = request.session.get("reset_email")

        if RegisteredUser.objects.filter(email=email).exists():
            user = RegisteredUser.objects.get(email=email)
            user.password = new_password  # Updating password
            user.save()
            messages.success(request, "Password reset successful! Please log in.")
            return redirect("user_login")

    return render(request, "reset_password.html")


from django.shortcuts import render
import zipfile
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb


from django.shortcuts import render
import zipfile, os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def training(request):

    # ---------------- CONFIG ----------------
    ZIP_PATH = "media/archive (1).zip"
    EXTRACT_PATH = "media/crypto_data"
    CRYPTO_FILE = "coin_Bitcoin.csv"
    SENTIMENT_FILE = "bitcoin_sentiment.csv"   # Date, Text
    TIME_STEPS = 90
    EPOCHS = 20
    BATCH_SIZE = 32

    # ---------------- UNZIP ----------------
    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(EXTRACT_PATH)

    # ---------------- LOAD PRICE DATA ----------------
    df = pd.read_csv(os.path.join(EXTRACT_PATH, CRYPTO_FILE))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # ---------------- LOAD SENTIMENT DATA ----------------
    sent_df = pd.read_csv(os.path.join(EXTRACT_PATH, SENTIMENT_FILE))
    sent_df['Date'] = pd.to_datetime(sent_df['Date'])

    analyzer = SentimentIntensityAnalyzer()
    sent_df["sentiment"] = sent_df["Text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    daily_sentiment = sent_df.groupby("Date")["sentiment"].mean().reset_index()

    # ---------------- MERGE ----------------
    df = pd.merge(df, daily_sentiment, on="Date", how="left")
    df["sentiment"] = df["sentiment"].fillna(0)

    # ---------------- FEATURE ENGINEERING ----------------
    df['HL'] = df['High'] - df['Low']
    df['OC'] = df['Open'] - df['Close']

    FEATURES = ['Close', 'Volume', 'HL', 'OC', 'Marketcap', 'sentiment']
    data = df[FEATURES].values

    # ---------------- SCALE ----------------
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, "media/scaler.save")

    # ---------------- CREATE SEQUENCES ----------------
    def create_sequences(data, steps):
        X, y = [], []
        for i in range(steps, len(data)):
            X.append(data[i-steps:i])
            y.append(data[i, 0])  # Close price
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, TIME_STEPS)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ---------------- LSTM MODEL ----------------
    input_layer = Input(shape=(TIME_STEPS, X.shape[2]))

    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    output_layer = Dense(1)(x)

    lstm_model = Model(inputs=input_layer, outputs=output_layer)
    lstm_model.compile(optimizer="adam", loss=tf.keras.losses.Huber())

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        "media/lstm_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    lstm_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=0
    )

    # ---------------- FEATURE EXTRACTION ----------------
    feature_model = Model(inputs=input_layer, outputs=x)
    X_train_feat = feature_model.predict(X_train)
    X_test_feat = feature_model.predict(X_test)

    # ---------------- XGBOOST ----------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(X_train_feat, y_train)
    joblib.dump(xgb_model, "media/xgboost_model.save")

    # ---------------- EVALUATION ----------------
    y_pred = xgb_model.predict(X_test_feat)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    price_accuracy = (1 - mape) * 100

    # ---------------- CONTEXT ----------------
    context = {
        "mape": round(mape, 4),
        "rmse": round(rmse, 4),
        "price_accuracy": round(price_accuracy, 2),
        "epochs": EPOCHS,
        "coin": "Bitcoin",
        "sentiment_used": "Yes"
    }

    return render(request, "training.html", context)


from django.shortcuts import render
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model, Model

def predict(request):
    TIME_STEPS = 90
    predicted_price = None

    if request.method == "POST":
        # 🔹 USER INPUT
        coin_file = request.POST.get("coin")

        DATA_PATH = f"media/crypto_data/{coin_file}"

        # LOAD DATA
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # FEATURE ENGINEERING
        df['HL'] = df['High'] - df['Low']
        df['OC'] = df['Open'] - df['Close']

        FEATURES = ['Close', 'Volume', 'HL', 'OC', 'Marketcap']
        data = df[FEATURES].values

        # LOAD SCALER
        scaler = joblib.load("media/scaler.save")
        data_scaled = scaler.transform(data)

        # LAST 90 DAYS
        last_sequence = data_scaled[-TIME_STEPS:]
        last_sequence = np.expand_dims(last_sequence, axis=0)

        # LOAD MODELS
        lstm_model = load_model("media/lstm_model.h5", compile=False)
        feature_layer_output = lstm_model.layers[-2].output
        feature_model = Model(inputs=lstm_model.input, outputs=feature_layer_output)

        lstm_features = feature_model.predict(last_sequence)

        xgb_model = joblib.load("media/xgboost_model.save")
        predicted_scaled_price = xgb_model.predict(lstm_features)[0]

        # INVERSE SCALE
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = predicted_scaled_price
        predicted_price = scaler.inverse_transform(dummy)[0, 0]

        last_price = df['Close'].iloc[-1]
        trend = "UP 📈" if predicted_price > last_price else "DOWN 📉"

        return render(request, "prediction.html", {
            "predicted_price": round(predicted_price, 2),
            "last_price": round(last_price, 2),
            "trend": trend
        })

    return render(request, "prediction.html")



