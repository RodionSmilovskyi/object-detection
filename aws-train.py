import boto3
import sagemaker
from sagemaker.estimator import Estimator

role = "arn:aws:iam::905418352696:role/SageMakerFullAccess"
boto_session = boto3.session.Session(profile_name="905418352696_AdministratorAccess", region_name="us-east-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)
my_region = boto_session.region_name
my_image_uri = (
    "763104351884.dkr.ecr."
    + my_region
    + ".amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker"
)

estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=my_image_uri,
    role=role,
    max_run=24 * 60 * 60,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    source_dir="container",
    entry_point="train.py"
)

estimator.fit(wait=False)
print("Done")