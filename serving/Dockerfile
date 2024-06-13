FROM --platform=linux/amd64 python:3.10.10-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN pip install .
COPY ./app.py ./app.py
COPY ./checkpoint.ckpt ./checkpoint.ckpt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]