# Expects datasets to be in /app/DroneTestDataset and /app/DroneTrainDataset

FROM tensorflow/tensorflow:1.2.1-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Setup NVidia repos.  This fixes the issue with a bad key from the NVidia base image.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/7fa2af80.pub



#RUN apt update
#RUN echo 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' >> /etc/apt/sources.list
#RUN echo 'deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' >> /etc/apt/sources.list
#RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA6932366A755776

RUN apt update
RUN apt install -y wget

RUN apt install -y build-essential libssl-dev libffi-dev python3-dev python3 python3-pip
RUN apt install -y protobuf-compiler python-pil python-lxml python-tk

# Install Python 3.7
RUN apt install -y build-essential checkinstall
RUN apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
RUN tar zxvf Python-3.7.9.tgz
WORKDIR /usr/src/Python-3.7.9
RUN ./configure --enable-optimizations
RUN make altinstall

WORKDIR /app

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install tensorflow

RUN apt install -y unzip

RUN apt install -y python-opencv python-pip nano
RUN python3.7 -m pip install pandas numpy
RUN python3.7 -m pip install Cython
RUN python3.7 -m pip install contextlib2
RUN python3.7 -m pip install jupyter
RUN python3.7 -m pip install matplotlib
RUN apt install -y git
RUN python3.7 -m pip install tensorflow-object-detection-api

#WORKDIR /usr/local/lib/python3.7/site-packages/tensorflow/
#RUN git clone https://github.com/tensorflow/models

#WORKDIR /app
#RUN git clone https://github.com/cocodataset/cocoapi.git
#WORKDIR /app/cocoapi/PythonAPI
#RUN sed -i 's/python/python3.7/g' Makefile
#RUN make

#RUN ls
#RUN cp -r pycocotools /usr/local/lib/python3.7/site-packages/tensorflow/models/research/

#WORKDIR /app
##RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v22.2/protoc-22.2-linux-x86_64.zip
#RUN wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
##RUN unzip protoc-22.2-linux-x86_64.zip
#RUN unzip protoc-3.3.0-linux-x86_64.zip
#RUN cp ./bin/protoc /usr/bin/protoc
#RUN protoc --version
#WORKDIR /usr/local/lib/python3.7/site-packages/tensorflow/models/research/
#RUN protoc object_detection/protos/*.proto --python_out=.
#RUN export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.7/site-packages/tensorflow/models/research/:/usr/local/lib/python3.7/site-packages/tensorflow/models/research/slim

RUN echo "Downloading scripts"
WORKDIR /app
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/RetrainModel.sh
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/xml_to_csv.py
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/generate_tfrecord.py

CMD sh RetrainModel.sh
