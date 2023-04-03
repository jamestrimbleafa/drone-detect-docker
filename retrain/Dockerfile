FROM tensorflow/tensorflow:1.2.1-gpu-py3
ENV FILEID=1FH-eifS6j8-Wo-ELcB-c1MdcdT3K5_HI
ENV FILENAME=PaperBasedANNModels.zip
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN apt update
RUN apt install -y wget
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
RUN apt install -y unzip
RUN unzip ${FILENAME} "droneInfGraph401092/*" -d "/app"
RUN apt install -y python3 python3-opencv python3-pip
RUN pip install numpy
RUN pip install tensorflow
RUN pip install pillow
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/main/src/RunModelImages.py

CMD python3 RunModelImages.py