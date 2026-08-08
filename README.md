# SimpleAutoML

Веб-система для обучения и выдачи моделей компьютерного зрения (классификация, сегментация). Стек: **React / Vite**, **FastAPI**, **Celery**, **PostgreSQL**, **Redis**, **MinIO**.

**Для разработки:** Python 3.10+, Node.js 18+ (сборка фронтенда), Docker с Compose V2.

## Содержание

- [Описание](#описание)
- [Автоматизация процессов](#автоматизация-процессов)
- [Загрузка данных](#загрузка-данных)
- [Тестирование (инференс)](#тестирование-инференс)
- [Локальный запуск](#локальный-запуск-проекта)
- [Docker и веб-интерфейс](#docker-и-веб-интерфейс)
- [Продакшен (чеклист)](#продакшен-чеклист)
- [Лицензия](#лицензия)
- [Структура репозитория](docs/PROJECT_LAYOUT.md)

## Описание

Автоматизирует подготовку данных, обучение и дообучение (YOLO через Ultralytics), инференс и хранение артефактов. Основной способ запуска — **Docker Compose** и единая точка входа через nginx.

**Схема веб-стека** (точка входа — `http://localhost:100`):

```mermaid
flowchart TB
  B(["Браузер"])
  subgraph compose["Docker Compose"]
    NG["nginx :100"]
    FE["frontend · React / Vite"]
    API["backend · FastAPI"]
    WRK["ml · Celery worker"]
    DB[("PostgreSQL")]
    RD[("Redis · брокер Celery")]
    MN["MinIO · S3 и консоль"]
    VOL[("том ml_data · /data")]
    B --> NG
    NG --> FE
    NG -->|"/api/"| API
    NG -->|"/minio/"| MN
    API --> DB
    API --> RD
    API -->|ZIP, Drive, выдача файлов| MN
    API --> VOL
    WRK --> RD
    WRK --> DB
    WRK --> VOL
    WRK -->|"POST /api/internal/storage/sync"| API
  end
```

### Поддерживаемые типы задач
1. **Классификация**  
   Определение категории, к которой принадлежит изображение.  
2. **Сегментация**  
   Пиксельная классификация изображения с разметкой объектов.

## Автоматизация процессов

Система автоматически анализирует предоставленный набор данных и определяет, какие действия необходимо выполнить:  
- **Обучение модели** — если данные загружены впервые.  
- **Дообучение модели** — если в данных обнаружены изменения или добавлены новые данные.  
- **Тестирование модели** — если в папке `test` присутствуют данные для инференса.  

### Управление моделями
Каждая обученная модель автоматически сохраняется в базе данных (БД), что позволяет:  
- Использовать уже обученные модели для быстрого тестирования.  
- Продолжать обучение без необходимости начинать с нуля.  

---

### Пример сценария использования
1. Загружаете новый набор данных — система автоматически обучает модель.  
2. Добавляете новые изображения в `dataset` — система выполняет дообучение.  
3. Загружаете изображения в папку `test` — система запускает инференс и сохраняет результаты.  

---

## Загрузка данных

Система поддерживает два способа загрузки данных: **Google Drive** и **ZIP-архив**.  

### **1. Загрузка через Google Drive**  
Настройте сервисный аккаунт Google (файл ключа задаётся в `SERVICE_ACCOUNT_FILE`, см. `.env.example`). В своём Google Drive необходимо:  
1. Создать директорию с вашим именем.  
2. Внутри этой директории создать папку с названием вашей задачи (например, `classification` или `segmentation`).  
3. Разместить в ней папку `dataset`, содержащую файлы для задачи.  

#### Структура данных для задач:
- **Классификация**  
  ```plaintext
  dataset/
  ├── class1/         # Директория для изображений первого класса
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class2/         # Директория для изображений второго класса
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
  ```
- **Сегментация**
    ```plaintext
    dataset/
    ├── images/         # Директория с изображениями
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── labels/         # Директория с разметкой
    │   ├── image1.txt  # Разметка для image1.jpg
    │   ├── image2.txt  # Разметка для image2.jpg
    │   └── ...
    ```
    Файл разметки для сегментации должен иметь структуру:
    ```plaintext
    класс: точки сегментов
    ```
### **2. Загрузка через ZIP-архив**
Архив загружается напрямую в директорию проекта и автоматически распаковывается.
#### Требования к структуре:
ZIP-архив должен содержать папку dataset с файлами, организованными так же, как описано выше для задач классификации и сегментации.

## Тестирование (инференс)

Для проведения инференса по вашей задаче необходимо:  
1. Создать в директории вашей задачи папку `test`.  
2. Разместить в папке `test` изображения, для которых требуется выполнить предсказание.  

После завершения инференса результаты сохранятся в папке `result` в директории вашей задачи.

### Пример структуры:
```plaintext
classification/            # Ваша задача
├── dataset/               # Данные для обучения
├── test/                  # Изображения для инференса
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── result/                # Результаты инференса (создается автоматически)
    ├── image1_result
    ├── image2_result
    └── ...
```

## Локальный запуск проекта

Для локального запуска проекта выполните следующие шаги:

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/OdincovMD/SimpleAutoML
   ```
2. Перейдите в директорию проекта:
    ```bash
    cd SimpleAutoML
    ```
3. Создайте виртуальное окружение:
    ```bash 
    python3 -m venv venv
    ```
4. Активируйте виртуальное окружение:
    - Для Linux/macOS:
    ```bash 
    source venv/bin/activate
    ```
    - Для Windows:
    ```bash 
    venv\Scripts\activate
    ```
5. Установите зависимости:
    - Worker / полный ML-стек локально: `pip install -r ml/requirements.txt`
    - Только backend (FastAPI): `pip install -r backend/requirements.txt`
6. Создайте файл `.env` в корне репозитория. Достаточно задать либо **`DATABASE_URL`** (строка подключения PostgreSQL), либо отдельные поля, как в `backend/config.py`:
    ```bash
    # Вариант A — одна строка (как в Docker)
    # DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/dbname

    # Вариант B — по полям
    DB_HOST=localhost
    DB_PORT=5432
    DB_USER=your_user
    DB_PASS=your_pass
    DB_NAME=your_db

    # Google Drive (опционально, для API)
    SERVICE_ACCOUNT_FILE=automl_token.json
    DRIVE_FOLDER_ID=your_drive_folder_id

    # Если поднимаете Celery worker и backend отдельно: общий секрет для POST /api/internal/storage/sync
    # INTERNAL_STORAGE_TOKEN=длинная-случайная-строка
    ```
7. **Таблицы PostgreSQL** создаются сами: при старте **FastAPI** (`SyncOrm.create_tables()` в [`backend/app/main.py`](backend/app/main.py)) и при **обучении в worker** (в начале [`run_pipeline`](backend/app/services/pipeline.py)). Отдельно инициализировать БД не нужно.

   Чтобы **полностью пересоздать** таблицы (все данные будут удалены):
   ```bash
   python -c "from backend.db.orm import SyncOrm; SyncOrm.init_db()"
   ```

## Docker и веб-интерфейс

### Требования

- Docker Engine и **Docker Compose V2**.

### Быстрый старт

```bash
cp .env.example .env
# В продакшене смените пароли и INTERNAL_STORAGE_TOKEN (одинаковый токен у backend и ml).
# Для Google Drive задайте DRIVE_FOLDER_ID и положите JSON ключа (см. SERVICE_ACCOUNT_FILE).
docker compose up -d --build
```

При старте контейнера **backend** схема БД в PostgreSQL создаётся автоматически (`SyncOrm.create_tables()` в lifespan приложения).

Точки входа после запуска:

| URL | Назначение |
|-----|------------|
| http://localhost:100 | Веб-портал (nginx → frontend) |
| http://localhost:100/api/ | REST API (FastAPI) |
| http://localhost:100/api/health | Проверка живости backend |
| http://localhost:100/minio/ | Консоль MinIO (прокси nginx → порт 9001 контейнера) |

### Состав стека

| Сервис | Роль |
|--------|------|
| **nginx** | Единая точка входа, лимит тела запроса 500M, прокси API и MinIO Console |
| **frontend** | SPA (React, Vite, TypeScript) |
| **backend** | FastAPI: датасеты, Drive, задачи, модели, работа с MinIO (S3 API) |
| **ml** | Celery worker: обучение YOLO; общий том `/data` с backend; после обучения вызывает внутренний API синхронизации в MinIO |
| **postgres** | Метаданные датасетов и моделей |
| **redis** | Брокер и backend результатов Celery |
| **minio** | Объектное хранилище (S3 API), том `minio_data` |

Healthcheck настроен для **nginx**, **backend**, **postgres**, **redis**, **ml**. У образа `minio/minio` отдельный healthcheck не используется (минимальный rootfs).

### Хранилище и безопасность

- С **MinIO по S3 API** взаимодействует только **backend** (загрузка ZIP в бакет, presigned URL для скачивания модели, выгрузка `results/` и `models/` после задачи).
- **ml** не содержит клиента MinIO: пишет артефакты на том **`ml_data`** (`/data` в контейнерах) и вызывает **`POST /api/internal/storage/sync`** с заголовком **`X-Internal-Token`**. Значение **`INTERNAL_STORAGE_TOKEN`** должно совпадать у сервисов **backend** и **ml** (задаётся в `.env` / compose; в примере по умолчанию в compose — только для разработки).

### Переменные окружения (Docker)

См. [`.env.example`](.env.example). Часто используемые:

- **`POSTGRES_*`** — учётные данные БД (и подстановка в `DATABASE_URL` у backend/ml).
- **`MINIO_ROOT_USER`**, **`MINIO_ROOT_PASSWORD`** — ключи MinIO (те же передаются backend как `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`).
- **`INTERNAL_STORAGE_TOKEN`** — секрет внутреннего API синхронизации с хранилищем.
- **`DATABASE_URL`**, **`CELERY_BROKER_URL`** — при необходимости переопределить явно.

Для **Google Drive** в контейнерах положите файл сервисного аккаунта в контекст сборки и смонтируйте или скопируйте в образ согласно вашей политике безопасности; в коде по умолчанию ожидается путь из **`SERVICE_ACCOUNT_FILE`** (`backend/config.py`).

### Обучение в контейнере

В **ml** по умолчанию заданы **`SKIP_IMGSZ_SEARCH=1`** и **`AUTO_IMGSZ=640`**, чтобы не запускать длительный перебор размера изображения (`check_imgsz`). Также **`AUTOML_QUIET=1`** отключает прогресс-бары tqdm в worker. Чтобы вернуть перебор `imgsz`, задайте **`SKIP_IMGSZ_SEARCH=0`**; для tqdm в логах — **`AUTOML_QUIET=0`**.

### Полезные команды

```bash
docker compose logs -f ml backend      # логи worker и API
docker compose build --no-cache ml     # пересборка после смены ml/requirements.txt
docker compose down -v                 # остановка и удаление томов (данные БД и MinIO пропадут)
```

## Продакшен (чеклист)

- **Секреты:** сильные `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, `INTERNAL_STORAGE_TOKEN`; не коммитьте `.env` и ключи (`automl_token.json` и т.п.).
- **CORS:** в [`backend/app/main.py`](backend/app/main.py) замените `allow_origins="*"` на список ваших доменов.
- **TLS:** вынесите HTTPS на внешний reverse proxy (Traefik, Caddy, облачный LB); не публикуйте HTTP-порт сервиса в открытую сеть без шифрования.
- **БД и бэкапы:** при старте backend создаёт таблицы и при необходимости добавляет колонки `task_type`, `trained_at`. Настройте резервное копирование PostgreSQL и томов MinIO по вашей политике.
- **Образы:** зафиксируйте теги образов (в т.ч. MinIO) вместо `latest`.
- **Frontend в CI:** `npm ci` и `npm run build` в `frontend/` для воспроизводимых артефактов.
- **Google Drive:** без валидного `DRIVE_FOLDER_ID` и сервисного аккаунта эндпоинты Drive не использовать; для продакшена предпочтительнее ZIP и MinIO.

## Лицензия

Этот проект распространяется под лицензией [MIT](LICENSE).

## Благодарности

Благодарим [@Horokami](https://github.com/Horokami), который внес свой вклад в развитие этого проекта. Также выражаем признательность за использование открытых библиотек и фреймворков, которые значительно ускоряют разработку. В частности, благодарим авторов библиотек:

[PyTorch](https://pytorch.org)  
[Google API](https://cloud.google.com/apis/)  
[ultralytics](https://github.com/ultralytics)

