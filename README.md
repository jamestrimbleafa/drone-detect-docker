# drone-detect-docker

There will be several versions of this Docker image targeting a desktop using CPU, desktop using GPU, raspberry pi using CPU, and raspberry pi using TPU.  Please download the Dockerfile for the version you'd like to use then follow the instructions below.  Note: Currently only the desktop CPU version is implemented.

## Building the Docker image
`docker build -t drone-detect .`

## Running the container

Place input images as .jpg or .png files in your source folder.  Output images will be written to the folder bound to "/app/out".

`docker run --mount type=bind,source=PATH-TO-YOUR-SOURCE-IMAGES,target=/app/in --mount type=bind,source=PATH-TO-YOUR-OUTPUT-DIRECTORY,target=/app/out drone-detect`