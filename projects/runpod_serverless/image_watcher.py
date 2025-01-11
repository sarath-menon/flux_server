from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import logging
import shutil
from pathlib import Path
import base64
import asyncio

import aiofiles
from PIL import Image
import io
from .image_watcher import ImageWatcher


class ImageWatcher(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue
        super().__init__()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            asyncio.run(self.queue.put(event.src_path))

async def watch_output_folder(queue):
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    event_handler = ImageWatcher(queue)
    observer = Observer()
    observer.schedule(event_handler, str(output_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        observer.stop()
    observer.join()