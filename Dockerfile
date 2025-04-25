# syntax=docker/dockerfile:1

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker

COPY src /

COPY aws-requirements.txt /

ENV PYTHONUNBUFFERED=TRUE

ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV WRAPT_DISABLE_EXTENSIONS=TRUE

ENV PATH="/:${PATH}"

ENV PYTHONPATH="/"

WORKDIR /

RUN pip install -r aws-requirements.txt

RUN chmod a+x train 

RUN rm -rf output

RUN rm -rf __pycache__