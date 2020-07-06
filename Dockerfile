FROM pytorch/pytorch:latest

RUN pip install --upgrade pip
# install opencv dependencies
RUN apt-get update && \
    apt-get install -y \
      `# opencv requirements` \
      libsm6 libxext6 libxrender-dev \
      `# opencv video opening requirements` \
      libv4l-dev

RUN apt-get install -y python-opencv

RUN mkdir -p /src/

WORKDIR /src/

COPY requirements.txt /src/

RUN pip install --no-cache-dir -r /src/requirements.txt

COPY . .
ENTRYPOINT python3 -u fakecam.py --device '/dev/video3' --hologram-effect
