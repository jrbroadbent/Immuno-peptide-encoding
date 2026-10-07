FROM python:3.12-bookworm 
COPY ./requirements.txt .
RUN apt-get update \
&& apt-get install python3-pip -y 
RUN pip install -r requirements.txt
