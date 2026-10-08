FROM python:3.12-bookworm 
RUN apt-get update \
&& apt-get install git -y 
RUN git clone https://github.com/jrbroadbent/Immuno-peptide-encoding.git
WORKDIR /Immuno-peptide-encoding
RUN pip install -r requirements.txt
