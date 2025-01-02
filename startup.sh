 #!/bin/bash

cd /home/flux_server/

# install dependencies for ai-toolkit
python -m pip install -r ai-toolkit/requirements.txt

# install poetry dependencies
poetry config virtualenvs.create false
poetry install

echo "$PWD"
export HF_HOME="/workspace/models"

cd src
python main.py
