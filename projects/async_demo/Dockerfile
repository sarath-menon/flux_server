FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL authors="sarath_suresh"

# Set up Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home
RUN git clone https://github.com/sarath-menon/flux_server.git

RUN pip install runpod

WORKDIR /

# Command to run with full path
CMD ["python", "/home/flux_server/development/runpod_workers/demo_async.py", "--rp_serve_api"]
