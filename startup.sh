 #!/bin/bash

cd /home/flux_server/

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc 

# install dependencies for ai-toolkit
python -m pip install -r flux_server/ai-toolkit/requirements.txt

# set ache directry for pip and huggingface to runpod network volume
pip config set global.cache-dir "/workspace/.cache/pip"
export HF_HOME="/workspace/models"

# install poetry dependencies
poetry config virtualenvs.create false
poetry install

echo "$PWD"


# start fastapi server
python projects/fastapi_server/main.py
