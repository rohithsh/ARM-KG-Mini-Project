FROM python:latest

WORKDIR /

COPY main.py main.py
COPY carcinogenesis.owl carcinogenesis.owl
COPY kg22-carcinogenesis_lps1-train.ttl kg22-carcinogenesis_lps1-train.ttl
COPY kg22-carcinogenesis_lps2-test.ttl kg22-carcinogenesis_lps2-test.ttl

RUN pip install rdflib rdfpandas pykeen sklearn numpy pandas torch

RUN python main.py