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
WORKDIR /workspace/flux_server

# Copy only requirements files first
COPY requirements.txt ai-toolkit/requirements.txt ./
COPY ai-toolkit/requirements.txt ./ai-toolkit/

# Copy the rest of the application
COPY . .
RUN chmod +x startup.sh

WORKDIR /

# Command to run with full path
CMD ["/bin/bash", "-c", "pwd && /workspace/flux_server/startup.sh"]