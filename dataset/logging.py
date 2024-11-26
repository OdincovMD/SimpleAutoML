import logging
import sys
import os

class StreamToLoggerAndStdout:
    """Перенаправляет вывод в логгер и сохраняет вывод в оригинальный stdout/stderr"""
    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream

    def write(self, buf):
        # Логирование строки
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)
        # Печать в оригинальный поток
        self.original_stream.write(buf)
        self.original_stream.flush()

    def flush(self):
        self.original_stream.flush()

def setup_logger():

    os.remove('app.log')
    # Настраиваем логгер
    logger = logging.getLogger()
    logger.handlers.clear()  # Удаляем старые обработчики
    logger.setLevel(logging.DEBUG)

    # Обработчик для записи логов в файл
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Перенаправляем stdout и stderr на логгер, сохраняя их в терминале
    sys.stdout = StreamToLoggerAndStdout(logger, logging.INFO, sys.__stdout__)  # INFO для stdout
    sys.stderr = StreamToLoggerAndStdout(logger, logging.ERROR, sys.__stderr__)  # ERROR для stderr