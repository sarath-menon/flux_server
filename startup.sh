 #!/bin/bash

cd /home/flux_server/

# install dependencies
python -m pip install -r requirements.txt && \
    python -m pip install -r ai-toolkit/requirements.txt

# huggingface-cli login --token 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

# echo "Downloading model weights"
# export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli download black-forest-labs/FLUX.1-dev
# echo "Finished downloadng model weights"

echo "$PWD"
export HF_HOME="/workspace/models"
python main.py