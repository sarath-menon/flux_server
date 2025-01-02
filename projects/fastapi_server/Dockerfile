FROM runpod/base:0.6.2-cuda12.2.0

LABEL authors="sarath_suresh"

# Install all system dependencies together
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    git-lfs \
    nvtop \
    htop

# Set up Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
# WORKDIR /home/flux_server
# COPY ../../ .
# Copy the application
WORKDIR /home
RUN git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules
RUN chmod +x /home/flux_server/projects/fastapi_server/startup.sh

WORKDIR /

# Command to run with full path
CMD ["/bin/bash", "-c", "pwd && /home/flux_server/projects/fastapi_server/startup.sh"]