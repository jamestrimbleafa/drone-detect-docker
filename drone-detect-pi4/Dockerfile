FROM ubuntu

WORKDIR /drone-detect
RUN apt-get update
RUN apt-get install libssl-dev openssl
wget https://www.python.org/ftp/python/3.5.0/Python-3.5.0.tgz
RUN apt-get install -y libatlas-base-dev

RUN pip3 install tensorflow
