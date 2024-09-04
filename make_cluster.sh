# comment
gcloud config set project exalted-beanbag-334502

# machine-type
# to see list of machines, do:
# $ gcloud compute machine-types list --filter="zone:(us-east4-b)"
# emily used "c2d-highcpu-4"

# image-type: The image type to use for the cluster. Defaults to server-specified.

# disk-size: Size for node VM boot disks in GB. Defaults to 100GB.

# metadata: Compute Engine metadata to be made available to the guest operating
# system running on nodes within the node pool.

# num-nodes: The number of nodes to be created in each of the cluster's zones.

gcloud container --project "exalted-beanbag-334502" clusters create "vasculature-features" \
    --zone "us-east4-b" \
    --no-enable-basic-auth \
    --release-channel "stable" \
    --machine-type "c2-standard-16" \
    --image-type "COS_CONTAINERD" \
    --disk-type "pd-standard" \
    --disk-size "100" \
    --metadata disable-legacy-endpoints=true \
    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --preemptible \
    --num-nodes "1" \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM \
    --enable-ip-alias \
    --network "projects/exalted-beanbag-334502/global/networks/patchseq" \
    --subnetwork "projects/exalted-beanbag-334502/regions/us-east4/subnetworks/patchseq" \
    --no-enable-intra-node-visibility \
    --default-max-pods-per-node "100" \
    --no-enable-master-authorized-networks \
    --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
    --enable-autoupgrade \
    --enable-autorepair \
    --max-surge-upgrade 3 \
    --max-unavailable-upgrade 0 \
    --enable-shielded-nodes \
    --node-locations "us-east4-b"

gcloud container clusters get-credentials --zone us-east4-b vasculature-features

# https://kubernetes.io/docs/concepts/configuration/secret/
kubectl create secret generic secrets \
    --from-file=$HOME/.cloudvolume/secrets/cave-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/google-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/discord-secret.json \