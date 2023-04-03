# Expects datasets to be in /app/DroneTestDataset and /app/DroneTrainDataset

FROM tensorflow/tensorflow:1.2.1-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN apt update
RUN apt install -y wget

RUN apt install -y unzip
RUN apt install -y python3-opencv python3-pip

RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/RetrainModel.sh
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/xml_to_csv.py
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/jt-retrain/retrain/src/generate_tfrecord.py

CMD sh RetrainModel.sh