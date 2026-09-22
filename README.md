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

## Hotfixing the runner without rebuilding the image

The runner script is baked into the image (`ADD . /app`), but `kube-task.yml` mounts a ConfigMap overlay on top of `runners/segclr_on_2026-09-09.py` so you can patch that one file without a rebuild/push. To apply a change:

1. Push the edited script into the ConfigMap (idempotent — re-run after every edit):

```sh
kubectl create configmap segclr-runner \
    --from-file=segclr_on_2026-09-09.py=runners/segclr_on_2026-09-09.py \
    --dry-run=client -o yaml | kubectl apply -f -
```

2. (First time only) apply the manifest so the volume/mount exists:

```sh
kubectl apply -f kube-task.yml
```

3. Restart the pods to pick up the new script:

```sh
kubectl rollout restart deployment/vasculature-features
```

> Notes: ConfigMaps are capped at 1 MiB. `subPath` mounts do not auto-refresh, so repeat steps 1 and 3 for each edit. This overlay causes the cluster to run a different script than the baked image — fold the change back into the image and rebuild once the fix is settled so the image stays the source of truth.
