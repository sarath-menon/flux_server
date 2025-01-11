 #!/bin/bash

cd /home/flux_server/

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
source ~/.bashrc 

# set cache directry for pip, huggingface to runpod network volume
pip config set global.cache-dir "/workspace/.cache/pip"
poetry config cache-dir "/workspace/.cache/pypoetry"
export HF_HOME="/workspace/models"

# # install dependencies for ai-toolkit
# python -m pip install -r flux_server/ai-toolkit/requirements.txt

# install poetry dependencies
poetry config virtualenvs.create false
poetry install

# # start fastapi server
# python projects/fastapi_server/main.py

# start runpod worker
python projects/runpod_serverless/rp_handler.py