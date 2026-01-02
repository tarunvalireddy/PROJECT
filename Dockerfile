FROM python:3.11.2

WORKDIR /APP

COPY requirements.txt /APP

RUN sudo apt install python3.11-venv \
    && python3 -m venv venv \
    && source venv/bin/activate \
    && pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD [ "python3","manage.py","runserver","0.0.0.0:8000" ]
    
