# Extracting features and labels for perivascular regions

To build:
`docker buildx build --platform linux/amd64 -t vasculature-features .`

To run:
`docker run --rm --platform linux/amd64 -v /Users/ben.pedigo/.cloudvolume/secrets:/root/.cloudvolume/secrets vasculature-features`

To tag:
`docker tag vasculature-features bdpedigo/vasculature-features:v0`

To push:
`docker push bdpedigo/vasculature-features:v0`

Making a cluster:
`sh ./make_cluster.sh`

Configuring a cluster:
`kubectl apply -f kube-task.yml`

Monitor the cluster:
`kubectl get pods`

Watch the logs in real-time:
`kubectl logs -f <pod-name>`
