# MSDS 631-02 - LLMs and MLOps course

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkthiebaut/guanaco)

Home of the LLM and MLOps course (MSDS 631-02 - University of San Francisco).

## Ray setup

- GCP Setup: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html
- AWS Setup: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html
- Getting started with Torch Lightning distributed: https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html

1. Setup the gcloud CLI tool (download and unzip the executable downloaded from [here](https://cloud.google.com/sdk/docs/install-sdk)):
2. `./google-cloud-sdk/install.sh`
3. `gcloud init`: create a new project
4. `gcloud auth application-default login`
5. Follow Ray’s [GCP cluster setup instructions](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html) (but replace the `project_id` with the name of the project created above.
6. Run the training script locally first (from the `ray` folder, `python guanaco_training_ray.py`) then on your Ray cluster (`ray submit ray-cluster-config.yml guanaco_training_ray.py --start`)

Troubleshooting: if you are unable to submit jobs to your cluster, you may have to edit your `/etc/ssh/ssh_config` file: `sudo vim /etc/ssh/ssh_config` → comment `SendEnv LANG LC\*\_` (i.e. replace with → `# SendEnv LANG LC\_\_` )
