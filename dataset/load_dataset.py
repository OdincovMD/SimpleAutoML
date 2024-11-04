from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm
import zipfile
import os
import io

def load_google_dataset():
    """
    Загружает файлы и папки из Google Диска в указанную локальную директорию,
    используя сервисный аккаунт для аутентификации. Пользователь вводит название
    папки, и функция рекурсивно скачивает все файлы и подпапки в эту папку.

    Возвращает:
        str: Название загруженной папки.
    """

    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = 'automl_token.json'

    credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=SCOPES
            )

    service = build(
        'drive', 
        'v3', 
        credentials=credentials
        )

    all_files = service.files().list(
        pageSize=10,
        fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
    folder_name = input(f"Папка с датасетом: {', '.join(
        folder['name'] for folder in all_files['files'] 
        if folder['mimeType'] == 'application/vnd.google-apps.folder'
        )}: ")

    for folder in all_files['files']:
        if folder['mimeType'] == 'application/vnd.google-apps.folder' and folder['name'] == folder_name:
            folder_id = folder['id']
            break
    else:
        raise FileNotFoundError(f'Вы ввели несуществующее название папки. Проверьте правильность названия и наличия папки: {folder_name}')

    def download_files_from_folder(folder_id, path='downloads'):
        """
        Скачивает все файлы из заданной папки на Google Диске в указанную локальную директорию.
        Работает рекурсивно, если обнаруживает подпапки.

        Аргументы:
            folder_id (str): Идентификатор папки на Google Диске.
            path (str): Локальный путь для сохранения файлов. По умолчанию 'downloads'.
        """

        if path and not os.path.exists(path):
            os.makedirs(path)

        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType, size)").execute()
        files = results.get('files', [])

        if not files:
            return

        for file in files:
            file_id = file['id']
            file_name = file['name']
            file_path = os.path.join(path, file_name)

            if file['mimeType'] == 'application/vnd.google-apps.folder':
                download_files_from_folder(file_id, file_path)
            else:
                request = service.files().get_media(fileId=file_id)
                fh = io.FileIO(file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)

                file_size = int(file.get('size', 0))
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        pbar.update(status.resumable_progress - pbar.n)


    download_files_from_folder(folder_id)

    return folder_name


def extract_zip(zip_path, extract_to='downloads'):
    """
    Извлекает содержимое zip-архива в указанную директорию.

    Аргументы:
        zip_path (str): Путь к zip-архиву (без расширения .zip).
        extract_to (str): Путь, куда будут извлечены файлы. По умолчанию 'downloads'.
    """

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path+'.zip', 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def main(load_type):
    """
    Основная функция, определяющая тип загрузки данных (Google Диск или zip-архив).
    В зависимости от параметра загружает файлы и папки с Google Диска или распаковывает
    zip-архив в локальную директорию.

    Аргументы:
        load_type (str): Тип загрузки, 'drive' для Google Диска и 'zip' для zip-архива.

    Возвращает:
        str: Название загруженной или извлеченной папки.

    Исключения:
        ValueError: Если передан некорректный тип загрузки.
    """
    
    if load_type == 'drive':
        folder = load_google_dataset()
    elif load_type == 'zip':
        folder = input('Название вашего zip-архива: ')
        extract_zip(folder)
    else:
        raise ValueError('Некоректный тип данных')
    return folder



