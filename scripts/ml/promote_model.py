"""Databricks entrypoint for manually promoting a trained model run."""

import argparse
import json
from dataclasses import asdict

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ml.nrt import DEFAULT_REGISTERED_MODEL
from madrid_ml.registry import promote_model_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--config-root", default="")
    parser.add_argument("--registered-model-name", default="")
    parser.add_argument("--model-alias", default="Champion")
    args = parser.parse_args()
    for name in ("model_run_id", "environment", "model_alias"):
        if not getattr(args, name).strip():
            parser.error(f"--{name.replace('_', '-')} must not be empty")
    return args


def main() -> int:
    args = parse_args()
    registered_model_name = args.registered_model_name.strip()
    if not registered_model_name:
        if not args.config_root.strip():
            raise ValueError("--config-root is required when resolving the model name")
        environment_config = ConfigLoader(args.config_root).load_environment(
            args.environment
        )
        registered_model_name = (
            f"{environment_config.catalogs.gold}.ml.{DEFAULT_REGISTERED_MODEL}"
        )
    result = promote_model_run(
        model_run_id=args.model_run_id,
        registered_model_name=registered_model_name,
        model_alias=args.model_alias,
    )
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    main()
