import logging
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class OutputFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.new_files = []
        
    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"New file created: {event.src_path}")
            self.new_files.append(event.src_path) 