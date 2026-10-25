"""Turn an MLflow model URI into something a human can act on.

Serving logs should answer "which model is this?" without anyone opening the
MLflow UI. Three URI shapes are handled:

    models:/PedestrianDetector@champion   -> "PedestrianDetector@champion (v3)"
    models:/PedestrianDetector            -> "PedestrianDetector (latest)"
    runs:/<run_id>/model                  -> "run:0322016"
    /path/to/model.pt                     -> "model"
"""

from __future__ import annotations


def describe_uri(uri: str) -> str:
    try:
        if uri.startswith("models:/"):
            rest = uri[len("models:/"):]
            if "@" in rest:
                name, alias = rest.rsplit("@", 1)
                version = _version_for_alias(name, alias)
                suffix = f" (v{version})" if version else ""
                return f"{name}@{alias}{suffix}"
            return f"{rest} (latest)"

        if uri.startswith("runs:/"):
            rest = uri[len("runs:/"):]
            run_id = rest.split("/", 1)[0]
            return f"run:{run_id[:7]}"
    except Exception:                                # noqa: BLE001
        pass
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _version_for_alias(name: str, alias: str) -> str | None:
    try:
        from mlflow.tracking import MlflowClient
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        return str(mv.version)
    except Exception:                                # noqa: BLE001
        return None
