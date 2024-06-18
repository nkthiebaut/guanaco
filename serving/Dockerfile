FROM --platform=linux/amd64 python:3.10.10-slim

COPY serving/requirements.txt ./requirements.txt
COPY ./setup.py ./setup.py

RUN pip install -r requirements.txt
RUN pip install .
COPY serving/app.py ./app.py
COPY serving/checkpoint.ckpt ./checkpoint.ckpt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]