# Extracting features and labels for perivascular regions

To build and push (single step):
`docker buildx build --platform linux/amd64 -t bdpedigo/vasculature-features:v0 --push .`

> Note: `docker buildx build` does not load the image into your local Docker store unless you pass `--load`, so `docker tag`/`docker push` on a plain build will fail with `No such image`. Building and pushing in one step (above) avoids this. If you need the image locally instead, add `--load` (single-platform only) and then tag/push manually.

To run:
`docker run --rm --platform linux/amd64 -v /Users/ben.pedigo/.cloudvolume/secrets:/root/.cloudvolume/secrets bdpedigo/vasculature-features:v0`

Making a cluster:
`sh ./make_cluster.sh`

Configuring a cluster:
`kubectl apply -f kube-task.yml`

Monitor the cluster:
`kubectl get pods`

Watch the logs in real-time:
`kubectl logs -f <pod-name>`
