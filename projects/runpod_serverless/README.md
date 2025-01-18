# Build docker image

```
./projects/runpod_serverless/build_docker.sh
```

# Testing

With fastapi server

```
python projects/runpod_serverless/rp_handler_async.py --rp_serve_api --rp_api_port 8006
```

With runpod worker

```
python projects/runpod_serverless/rp_handler_async.py
```

Run client

```
python projects/runpod_serverless/test_client.py
```

Copy sample image to output directory

```
rm -rf output/sample.png
cp resources/sample.png output/sample.png
```
