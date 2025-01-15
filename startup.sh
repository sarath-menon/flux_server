 #!/bin/bash

echo "Running startup.sh"
cd /home/flux_server/

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
source ~/.bashrc 

# set cache directry for pip, huggingface to runpod network volume
pip config set global.cache-dir "/workspace/.cache/pip"
poetry config cache-dir "/workspace/.cache/pypoetry"
export HF_HOME="/workspace/models"

# install poetry dependencies
poetry config virtualenvs.create false
poetry install

# # start runpod worker
# python projects/runpod_serverless/rp_handler.py