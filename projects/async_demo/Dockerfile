FROM runpod/base:0.6.2-cpu

LABEL authors="sarath_suresh"

# Set up Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home
RUN git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules

WORKDIR /

# Command to run with full path
CMD ["python", "/home/flux_server/development/runpod_workers/demo_async.py", "--rp_serve_api"]
