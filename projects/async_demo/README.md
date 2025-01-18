### Docker build

```
docker build -t runpod_async_demo projects/async_demo
docker tag runpod_async_demo sarathmenon1999/runpod_async_demo:0.01
docker push sarathmenon1999/runpod_async_demo:0.01
```

Request body

```
{
  "input": {
    "number": "7"
  }
}
```
