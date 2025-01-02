 #!/bin/bash

cd /home/flux_server/

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc 

# install dependencies for ai-toolkit
python -m pip install -r flux_server/ai-toolkit/requirements.txt

# install poetry dependencies
poetry config virtualenvs.create false
poetry install

echo "$PWD"
export HF_HOME="/workspace/models"

# start fastapi server
python projects/fastapi_server/main.py
