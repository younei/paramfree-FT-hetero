# base image of ubuntu OS 
FROM ubuntu:18.04 

LABEL maintainer "Youning Xia" 

# install python 
RUN apt update \
    && apt install -y python3-pip

# specify working directory in the container 
COPY . /project
WORKDIR /project 

# install dependencies 
RUN pip3 install -r requirements.txt




