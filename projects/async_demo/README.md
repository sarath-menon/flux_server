### Docker build

```
docker build -t runpod_async_demo projects/async_demo
docker tag runpod_async_demo sarathmenon1999/runpod_async_demo:0.01
docker push sarathmenon1999/runpod_async_demo:0.01
```

### Runpod serverless

```
runpod-serverless deploy --function-name runpod_async_demo --handler development/runpod_workers/demo_async.py --name runpod_async_demo --region us-east-1 --memory 1024 --timeout 300 --concurrency 10
```
