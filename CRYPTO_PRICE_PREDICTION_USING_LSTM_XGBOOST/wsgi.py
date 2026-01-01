"""
WSGI config for CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CRYPTO_PRICE_PREDICTION_USING_LSTM_XGBOOST.settings')

application = get_wsgi_application()
