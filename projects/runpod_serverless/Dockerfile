FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL authors="sarath_suresh"

# Install all system dependencies together
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    git-lfs 

# Set up Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install runpod

WORKDIR /home
RUN git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules
RUN chmod +x ./flux_server/projects/runpod_serverless/startup.sh

WORKDIR /

# Command to run with full path
CMD ["/bin/bash", "-c", "pwd && /home/flux_server/projects/runpod_serverless/startup.sh"]