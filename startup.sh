 #!/bin/bash

cd /home/flux_server/

# install dependencies
python -m pip install -r requirements.txt && \
    python -m pip install -r ai-toolkit/requirements.tx

echo "$PWD"
export HF_HOME="/workspace/models"
python main.py
