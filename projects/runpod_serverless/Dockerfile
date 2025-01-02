FROM runpod/base:0.6.2-cuda12.2.0

LABEL authors="sarath_suresh"

# Install all system dependencies together
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    git-lfs 

# Set up Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install runpod

# Set working directory
WORKDIR /home/flux_server

# Copy the application
COPY . .
RUN chmod +x startup.sh

WORKDIR /

# Command to run with full path
CMD ["/bin/bash", "-c", "pwd && /home/flux_server/startup.sh"]