"""Подавление «консольного» вывода (tqdm и т.п.) в worker / Docker."""
import os


def tqdm_disable() -> bool:
    return os.environ.get("AUTOML_QUIET", "1").lower() in ("1", "true", "yes")
