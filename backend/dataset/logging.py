import logging
import sys
import os

class StreamToLoggerAndStdout:
    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)
        self.original_stream.write(buf)
        self.original_stream.flush()

    def flush(self):
        self.original_stream.flush()

def setup_logger():
    if os.path.exists('app.log'):
        os.remove('app.log')        
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    sys.stdout = StreamToLoggerAndStdout(logger, logging.INFO, sys.__stdout__)
    sys.stderr = StreamToLoggerAndStdout(logger, logging.ERROR, sys.__stderr__)
