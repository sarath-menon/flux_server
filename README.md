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

# Testing

## Fastapi server 

Sample CURL request 
```
curl -X POST "http://localhost:8000/train" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./data.zip" \
  -F 'params={"trigger_word":"TOK","autocaption":true,"steps":1000,"learning_rate":0.0004,"batch_size":1,"resolution":"512,768,1024","lora_rank":16,"caption_dropout_rate":0.05,"optimizer":"adamw8bit"}'
```


## Runpod workser (server)

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