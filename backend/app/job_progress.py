"""Идентификаторы этапов обучения (meta Celery, API)."""

# Порядок шагов для прогресса (в одном прогоне не все встречаются)
STEP_ORDER = [
    "job_queued",
    "indexing_files",
    "splitting",
    "learning",
    "saving_model",
    "fine_tune_split",
    "fine_tuning",
    "fine_tune_saved",
    "nothing_to_train",
    "saving_to_cloud",
    "job_complete",
]
