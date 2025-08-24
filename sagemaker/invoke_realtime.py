import argparse, sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-artifact", required=True)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--output-s3", required=True, help="S3 prefix for async results")
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--role", default=None)
    p.add_argument("--framework-version", default="2.4")
    p.add_argument("--py-version", default="py311")
    args = p.parse_args()

    sess = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()

    model = PyTorchModel(
        model_data=args.model_artifact,
        role=role,
        entry_point="inference.py",
        source_dir="sagemaker",
        framework_version=args.framework_version,
        py_version=args.py_version,
        dependencies=["src"],
        sagemaker_session=sess,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        async_inference_config=AsyncInferenceConfig(output_path=args.output_s3),
    )
    print("Async endpoint:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
