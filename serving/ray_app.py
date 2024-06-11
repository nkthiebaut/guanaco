import requests
from fastapi import FastAPI
from ray import serve
from loguru import logger

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    @app.get("/")
    def get_root(self) -> dict:
        logger.info("Received request on the root endpoint")
        return {"status": "ok"}


ray_app = FastAPIDeployment.bind()
serve.run(ray_app, route_prefix="/")

print(requests.get("http://localhost:8000/").json())

# Docs: https://docs.ray.io/en/latest/serve/develop-and-deploy.html

# Serve locally with: ray serve ray_app:ray_app
# See app docs at: http://localhost:8000/docs

# Build serving config files: serve build ray_app:ray_app -o ray-config.yaml
# Deploy the config locally with: serve run config.yaml

# To deploy your API to production:
# Docs: https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#serve-in-production-deploying
# Run serve deploy config_file.yaml -a http://{your_cluster_ip}:8265
