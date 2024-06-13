# MSDS 631-02 - LLMs and MLOps course

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkthiebaut/guanaco)

Home of the LLM and MLOps course (MSDS 631-02 - University of San Francisco).

## Ray setup

1. Setup the gcloud CLI tool (download and unzip the executable downloaded from [here](https://cloud.google.com/sdk/docs/install-sdk)):
2. `./google-cloud-sdk/install.sh`
3. `gcloud init`: create a new project
4. `gcloud auth application-default login`
5. Follow Ray’s [GCP cluster setup instructions](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html) (but replace the `project_id` with the name of the project created above.
6. Run the training script locally first (from the `ray` folder, `python guanaco_training_ray.py`)
7. Create a Google Storage bucket to store your runs: `gcloud storage buckets create gs://model-repo-msds-631-02 --location=US-WEST1`, then a folder in the above bucket: `gcloud storage managed-folders create gs://model-repo-msds-631-02/training_runs`
8. Run the training script on your Ray cluster (`ray submit ray-cluster-config.yml guanaco_training_ray.py --start`)

Troubleshooting: if you are unable to submit jobs to your cluster, you may have to edit your `/etc/ssh/ssh_config` file: `sudo vim /etc/ssh/ssh_config` → comment `SendEnv LANG LC\*\_` (i.e. replace with → `# SendEnv LANG LC\_\_` )

### Links

- GCP Setup: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html
- AWS Setup: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html
- Getting started with Torch Lightning distributed: https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html

## Deploy your model

(This section mainly follows [this guide](https://github.com/anyscale/academy/tree/main/ray-serve/e2e/deploy-cloud-run))

Two options: use [Anyscale](https://github.com/anyscale/academy/blob/main/ray-serve/e2e/deploy-cloud-run/anyscale.com) (as of June '24 they offer $50 in credits pre sign-up), or GCP. Let's do the latter for simplicity.

From the `serving` folder:

- Create a GCP repository for the project: `gcloud artifacts repositories create guanaco-containers-repo --repository-format=docker --location=us --description="My containers repo for serving"` (or follow these instructions: https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo)
- Set Google Cloud Registry (GCR) as Docker Hub: `gcloud auth configure-docker`
- Edit the model path in GS, then run `./download_model.sh` to include the trained model in the Docker image.
- Build the serving image: `docker build --platform=linux/amd64 -t guanaco-containers-repo/guanaco-app .`
- Tag and push your image to GCR: `docker tag guanaco-containers-repo/guanaco-app gcr.io/msds-631-02/guanaco-app && docker push gcr.io/msds-631-02/guanaco-app`

Finally, deploy your app container with the following:

```bash
gcloud run deploy guanaco-app \
    --image=gcr.io/msds-631-02/guanaco-app \
    --allow-unauthenticated \
    --port=80 \
    --concurrency=80 \
    --cpu=4 \
    --memory=8192Mi \
    --platform=managed \
    --region=us-central1 \
    --project=msds-631-02
```

The command will print an output URL for your service (similar to `https://guanaco-app-etmhkt2fga-uc.a.run.app`). You can now send requests to your API (e.g. with the swagger interface: `https://guanaco-app-etmhkt2fga-uc.a.run.app/docs`)!
