FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL authors="sarath_suresh"
# Install all system dependencies together
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    git-lfs 

RUN pip install runpod

# symlink runpod volume to workspace
RUN rm -rf /workspace && ln -s /runpod-volume /workspace

# Add a build argument for cache busting
ARG CACHEBUST=1

WORKDIR /home
# Use the build argument to invalidate the cache
RUN git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules && echo $CACHEBUST
RUN chmod +x ./flux_server/projects/runpod_serverless/startup.sh

WORKDIR /

# Command to run with full path
CMD ["/bin/bash", "-c", "pwd && /home/flux_server/projects/runpod_serverless/startup.sh"]