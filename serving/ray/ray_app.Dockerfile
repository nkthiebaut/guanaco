FROM rayproject/ray:latest
EXPOSE 8000
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY ./ray_app.py ./ray_app.py
COPY ./ray-config.yaml ./ray-config.yaml

CMD serve run ray-config.yaml