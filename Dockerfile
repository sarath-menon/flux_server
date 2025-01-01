FROM runpod/base:0.6.2-cuda12.2.0

LABEL authors="sarath_suresh"

# Install dependencies
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx  git-lfs

WORKDIR /workspace

ARG CACHEBUST=1
RUN git clone --recursive https://github.com/sarath-menon/flux_server
WORKDIR /workspace/flux_server
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt
RUN python -m pip install -r ai-toolkit/requirements.txt

RUN apt-get install -y tmux nvtop htop

CMD ["python" "server.py"]
