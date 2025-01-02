# Flux server

## Install
```
git clone https://github.com/sarath-menon/flux_server.git --recurse-submodules
chmod +x startup.sh 
./startup.sh
```

# Docker build

## Fastapi server 
```
docker build -t flux_server_fastapi projects/fastapi_server
docker tag flux_server_fastapi sarathmenon1999/flux_server_fastapi:0.01
docker push sarathmenon1999/flux_server_fastapi:0.01
```

## Runpod worker
```
docker build -t flux_server_runpod projects/runpod_server
docker tag flux_server_runpod sarathmenon1999/flux_server_runpod:0.01
docker push sarathmenon1999/flux_server_runpod:0.01
```

```
docker builder prune
```