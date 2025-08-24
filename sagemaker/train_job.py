import argparse, time, yaml, sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sagemaker/config.example.yaml")
    p.add_argument("--job-name", default=None)
    p.add_argument("--role", default=None)
    p.add_argument("--use-spot", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sess = sagemaker.Session()
    role = args.role or cfg.get("role_arn") or sagemaker.get_execution_role()

    tr = cfg.get("training", {})
    estimator = PyTorch(
        entry_point="train_emotts.py",   # <— your script at repo root
        source_dir=".",                  # <— installs root requirements.txt for training
        dependencies=["src"],            # package your library code
        framework_version=str(tr.get("framework_version", "2.4")),
        py_version=str(tr.get("py_version", "py311")),
        instance_type=tr.get("instance_type", "ml.g5.2xlarge"),
        instance_count=int(tr.get("instance_count", 1)),
        role=role,
        hyperparameters=cfg.get("hyperparameters", {}),
        distribution=tr.get("distribution", None),
        sagemaker_session=sess,
        enable_sagemaker_metrics=True,
        use_spot_instances=args.use_spot,
        max_wait=7200 if args.use_spot else None,
        max_run=7200,
    )

    # Map named data channels (become SM_CHANNEL_<NAME> in the container)
    inputs = {n: TrainingInput(uri) for n, uri in cfg.get("data_channels", {}).items()}
    job_name = args.job_name or f"emotts-{int(time.time())}"
    estimator.fit(inputs if inputs else None, job_name=job_name)
    print("Training job:", job_name)
    print("Model artifacts:", estimator.model_data)

if __name__ == "__main__":
    main()
