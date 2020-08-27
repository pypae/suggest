FROM python:3.8

RUN pip install pip install transformers[torch] jellyfish fastapi[all]

WORKDIR /app

COPY suggest /app

CMD uvicorn --host 0.0.0.0 --port 80 main:app