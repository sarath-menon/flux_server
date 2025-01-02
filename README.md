# Flux server

## Install
```
git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules
cd flux_server
chmod +x startup.sh 
./startup.sh
```
# Projects

## Fastapi server 

### Docker build 
```
docker build -t flux_server_fastapi projects/fastapi_server
docker tag flux_server_fastapi sarathmenon1999/flux_server_fastapi:0.01
docker push sarathmenon1999/flux_server_fastapi:0.01
```

## Runpod worker

### Docker build 
```
docker build -t flux_server_runpod projects/runpod_server
docker tag flux_server_runpod sarathmenon1999/flux_server_runpod:0.01
docker push sarathmenon1999/flux_server_runpod:0.01
```

## Testing
```
python projects/runpod_serverless/rp_handler.py
```

With fastapi server
```
python projects/runpod_serverless/rp_handler.py --rp_serve_api
```

# Misc
```
docker builder prune
```

set pip cache dir
```
pip config set global.cache-dir "/workspace/.cache/pip"
```