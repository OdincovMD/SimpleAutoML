# Структура репозитория SimpleAutoML

Краткая карта каталогов и сервисов.

## Корень

| Путь | Назначение |
|------|------------|
| `docker-compose.yml` | Веб-стек: nginx, frontend, backend, ml (Celery), Postgres, Redis, MinIO |
| `nginx/` | Прокси: `/` → frontend, `/api/` → backend |
| `pyproject.toml` | Метаданные пакета; зависимости по сервисам — в `backend/requirements.txt`, `ml/requirements.txt` |
| `README.md` | Запуск, сценарии, переменные окружения |

## Backend (`backend/`)

| Путь | Назначение |
|------|------------|
| `app/main.py` | FastAPI: роутеры, lifespan; при старте `ALTER` колонок `task_type`/`trained_at` (PG), `create_tables`, MinIO buckets |
| `app/api/` | HTTP: датасеты, Drive, модели, инференс, jobs, internal storage |
| `app/services/` | Бизнес-логика без HTTP: `pipeline.py`, `storage.py`, `drive.py` (листинг/скачивание для API) |
| `app/tasks.py` | Celery: обучение, инференс, вызовы internal API |
| `app/job_progress.py` | Коды этапов для прогресса задач |
| `config.py` | Pydantic Settings (БД, Redis, MinIO, Drive, пути) |
| `db/` | SQLAlchemy модели и синхронный ORM |
| `dataset/` | Разбиение данных, тип задачи, логирование |
| `integrations/` | Вызовы внешних API вне HTTP (загрузка результатов в Google Drive из worker) |
| `exception/` | Исключения домена |

## ML worker (`ml/`)

Код Ultralytics/YOLO: `model.py`, аугментации, проверка `imgsz`. Импортирует `backend.*` при запуске worker из образа с `PYTHONPATH=/app`.

## Frontend (`frontend/`)

Vite + React: `src/pages/` (Upload, Jobs, Models, Inference), `src/api.ts`, общие UI в `src/components/ui/`.

## Потоки данных (кратко)

1. **Обучение:** ZIP или Drive → backend → MinIO `datasets/` → Celery `train_task` → диск `/data` → POST internal sync → MinIO `models/` / `results/`.
2. **Скачивание весов:** GET `/api/models/{id}/weights` — поток из MinIO через backend.
3. **Инференс:** ZIP → Celery → internal upload → GET `/api/inference/.../results/...`.
