import argparse, os, sagemaker, boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.model_monitor import DataCaptureConfig

def _model_artifact_from_job(sm, job):
    desc = sm.describe_training_job(TrainingJobName=job)
    return desc["ModelArtifacts"]["S3ModelArtifacts"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-artifact", help="s3://.../model.tar.gz")
    p.add_argument("--training-job", help="If set, pull the artifact from this job")
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--role", default=None)
    p.add_argument("--framework-version", default="2.4")
    p.add_argument("--py-version", default="py311")
    p.add_argument("--capture-s3", default=None, help="s3://... to enable DataCapture")
    args = p.parse_args()

    sess = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    sm = boto3.client("sagemaker", region_name=sess.boto_region_name)

    model_data = args.model_artifact or _model_artifact_from_job(sm, args.training_job)

    model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir="sagemaker",       # packs handler + sagemaker/requirements.txt into `code/`
        framework_version=args.framework_version,
        py_version=args.py_version,
        dependencies=["src"],
        sagemaker_session=sess,
    )

    capture = None
    if args.capture_s3:
        capture = DataCaptureConfig(enable_capture=True, sampling_percentage=100, destination_s3_uri=args.capture_s3)

    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        data_capture_config=capture,
    )
    print("Endpoint:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
