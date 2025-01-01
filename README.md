# Flux server

## Docker build

```
docker build -t flux_server .
```

```
docker run -d --name flux_server -p 8082:8082 clicking_server
docker tag flux_server sarathmenon1999/flux_server:0.4
docker push sarathmenon1999/flux_server:0.4
```

```
docker builder prune
```