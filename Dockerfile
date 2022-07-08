FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workdir
COPY . ./

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

ENTRYPOINT uvicorn app:app --host 0.0.0.0
