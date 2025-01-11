#!/bin/bash

# Read version from version.txt or create if doesn't exist
VERSION_FILE="version.txt"
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(cat $VERSION_FILE)
else
    VERSION="0.0.1"
    echo $VERSION > $VERSION_FILE
fi

# Build and push Runpod worker
docker build -t flux_server_runpod --build-arg CACHEBUST=$(date +%s) projects/runpod_serverless
docker tag flux_server_runpod sarathmenon1999/flux_server_runpod:${VERSION}
docker push sarathmenon1999/flux_server_runpod:${VERSION}

# Increment patch version for next build
NEXT_VERSION=$(echo $VERSION | awk -F. '{$NF = $NF + 1;} 1' | sed 's/ /./g')
echo $NEXT_VERSION > $VERSION_FILE

echo "Successfully built and pushed version ${VERSION}"
echo "Next version will be ${NEXT_VERSION}" 