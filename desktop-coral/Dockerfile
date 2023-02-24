FROM ubuntu
ENV FILEID=1FH-eifS6j8-Wo-ELcB-c1MdcdT3K5_HI
ENV FILENAME=PaperBasedANNModels.zip
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
# Install the Google Coral Coral Repository
RUN apt install -y wget curl
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update Apt and install packages
RUN apt update
RUN apt install libedgetpu1-std
RUN apt install -y python3 python3-opencv python3-pip
RUN apt install python3-pycoral
RUN pip install numpy
RUN pip install tensorflow
RUN pip install pillow

# Download model and python scripts
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
RUN apt install -y unzip
RUN unzip ${FILENAME} "droneInfGraph401092/*" -d "/app"
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/main/src/RunModelImages.py
RUN wget https://raw.githubusercontent.com/jamestrimbleafa/drone-detect-docker/main/src/RunModelImages_lite.py

# Convert the TFv1 model to TF-lite
RUN python3 TF1-TFLiteConversion.py

CMD python3 RunModelImages.py