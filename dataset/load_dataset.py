from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from tqdm import tqdm
from src.queries.orm import SyncOrm
from exception.file_system import FolderError, EmptyFolderError, DownloadTypeError, DownloadError
import zipfile
import os
import io
import traceback


def load_google_dataset():
    """
    Загружает файлы и папки из Google Диска в указанную локальную директорию,
    используя сервисный аккаунт для аутентификации. Пользователь вводит название
    папки, и функция рекурсивно скачивает все файлы и подпапки в эту папку.

    Возвращает:
        str: Название загруженной папки.
    """
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive']
        SERVICE_ACCOUNT_FILE = 'automl_token.json'

        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
        
        service = build(
            'drive',
            'v3', 
            credentials=credentials
            )

        user_folder = input('Название вашей папки: ')

        all_files = service.files().list(
            q=f"name='{user_folder}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()

        if not all_files['files']:
            raise FolderError(user_folder)
        
        parent_folder_id = all_files['files'][0]['id']

        target_folders = service.files().list(
            q=f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute().get('files', [])

        if not target_folders:
            raise EmptyFolderError(user_folder)

        folder_name = input(f"Папка с датасетом {'| '.join(folder['name'] for folder in target_folders)}: ")

        for folder in target_folders:
            if folder['name'] == folder_name:
                folder_id = folder['id']
                break
        else:
            raise FolderError(folder_name)

        def get_all_files_in_folder(folder_id):
            """
            Рекурсивно получает список всех файлов в папке и её подпапках на Google Диске.
            """
            files = []
            query = f"'{folder_id}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            folder_files = results.get('files', [])

            for file in folder_files:
                # Пропускаем обработку, если папка называется 'result'
                if file['mimeType'] == 'application/vnd.google-apps.folder' and file['name'] == 'result':
                    continue
                if file['mimeType'] == 'application/vnd.google-apps.folder':
                    # Рекурсивно обрабатываем подпапку и добавляем путь к папке
                    subfolder_files = get_all_files_in_folder(file['id'])
                    for subfile in subfolder_files:
                        subfile['path'] = os.path.join(file['name'], subfile['path'])
                    files.extend(subfolder_files)
                else:
                    # Добавляем информацию о файле с базовым путём (без папок)
                    files.append({'id': file['id'], 'name': file['name'], 'path': file['name']})

            return files

        def download_files_from_folder(folder_id, download_path=f'downloads/{user_folder}/{folder_name}'):
            """
            Скачивает все файлы из заданной папки на Google Диске в указанную локальную директорию.
            Работает рекурсивно, если обнаруживает подпапки. Пропускает файлы, которые уже есть в базе данных.

            Аргументы:
                service: Авторизованный объект службы Google Drive API.
                folder_id (str): Идентификатор папки на Google Диске.
                folder_name (str): Название папки в базе данных.
                download_path (str): Локальный путь для сохранения файлов. По умолчанию 'downloads'.
            """
            # Получаем список уже загруженных файлов из базы данных
            old_files = [el[0] for el in SyncOrm.select_data(download_path)] 
            all_files = get_all_files_in_folder(folder_id)
            total_files = len(all_files)

            if not all_files:
                print("Нет файлов для скачивания.")
                return

            with tqdm(total=total_files, desc="Скачивание файлов", unit="файл") as file_pbar:
                for file in all_files:
                    file_id = file['id']
                    file_name = file['name']
                    file_relative_path = os.path.join(download_path, file['path'])
                    # Пропускаем файл, если он уже есть в базе данных
                    if file_relative_path in old_files:
                        file_pbar.update(1)
                        continue

                    # Создаём подпапки для файла, если они ещё не существуют
                    if not os.path.exists(os.path.dirname(file_relative_path)):
                        os.makedirs(os.path.dirname(file_relative_path))

                    # Загружаем файл
                    request = service.files().get_media(fileId=file_id)
                    with io.FileIO(file_relative_path, 'wb') as fh:
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    file_pbar.update(1)

        download_files_from_folder(folder_id)

        return f'downloads/{user_folder}/{folder_name}'
    except Exception as error:
        raise DownloadError(traceback.format_exc())
    
def upload_to_drive(filepath, drive_path):
    """
    Загружает файл на Google Диск по заданному пути.

    Параметры:
        filepath (str): Путь к файлу для загрузки (локальный файл)
        drive_path (str): Путь к папке на Google Диске (например, 'folder1/subfolder2')
    """
    SERVICE_ACCOUNT_FILE = 'automl_token.json'

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
        )
    service = build('drive', 'v3', credentials=creds)

    folder_id = '1tltCIfYpj28-xbc3Vzc4-CgXRxF2KAsU'
    for folder_name in drive_path.split('/'):
        print(f"Обрабатываю папку: {folder_name}")
        query = f"'{folder_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])
        
        if not files:
            print(f"Папка '{folder_name}' не найдена. Создаю новую...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folder_id],
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            print(f"Создана папка '{folder_name}' с ID: {folder_id}")
        else:
            folder_id = files[0]['id']
            print(f"Папка '{folder_name}' найдена с ID: {folder_id}")

    file_metadata = {'name': filepath.split('/')[-1], 'parents': [folder_id]}
    media = MediaFileUpload(filepath)
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Файл загружен. ID файла: {uploaded_file.get('id')}")

def extract_zip(zip_path, extract_to='downloads/'):
    """
    Извлекает содержимое zip-архива в указанную директорию.

    Аргументы:
        zip_path (str): Путь к zip-архиву (без расширения .zip).
        extract_to (str): Путь, куда будут извлечены файлы. По умолчанию 'downloads'.
    """
    user_folder = input('Введите название вашей папки: ')
    user_task = input('Введите название вашей задачи: ')
    path = os.path.join(extract_to, user_folder, user_task)
    if not os.path.exists(path):
        os.makedirs(path)
    
    with zipfile.ZipFile(zip_path+'.zip', 'r') as zip_ref:
        zip_ref.extractall(path)
    return path


def main():
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
    load_type = input('Тип загрузки drive | zip: ')
    if load_type == 'drive':
        folder = load_google_dataset()
    elif load_type == 'zip':
        folder = extract_zip(input('Название вашего zip-архива: '))
    else:
        raise DownloadTypeError(load_type)
    return folder