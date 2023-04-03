# Expects datasets to be in /app/DroneTestDataset and /app/DroneTrainDataset

FROM tensorflow/tensorflow:1.2.1-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Setup NVidia repos.  This fixes the issue with a bad key from the NVidia base image.
#RUN apt update
#RUN apt-get -y install apt-transport-https
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/7fa2af80.pub

RUN apt update
# RUN apt install -y apt-transport-https ca-certificates
RUN apt install -y wget

RUN apt install -y unzip
RUN apt install -y python-opencv python-pip
RUN pip install pandas, numpy

RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/RetrainModel.sh
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/xml_to_csv.py
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/generate_tfrecord.py

CMD sh RetrainModel.sh
