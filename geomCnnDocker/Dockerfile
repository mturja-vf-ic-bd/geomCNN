FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
ARG USER_ID
ARG GROUP_ID

RUN apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENV HOME=/home/user
RUN chown -R $USER "$Turja_HOME" $HOME

RUN mkdir $HOME/geomCNN
COPY requirements.txt $HOME/geomCNN
WORKDIR $HOME/geomCNN
RUN pip3 install -r requirements.txt
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
