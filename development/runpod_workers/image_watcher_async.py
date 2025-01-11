import runpod
import asyncio
import os
import base64
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            self.queue.put(event.src_path)

async def async_generator_handler(job):
    output_dir = "./output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up queue and file watcher
    image_queue = Queue()
    event_handler = ImageEventHandler(image_queue)
    observer = Observer()
    observer.schedule(event_handler, output_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            # Check for new images
            if not image_queue.empty():
                image_path = image_queue.get()
                filename = os.path.basename(image_path)
                
                # Read and encode the image
                try:
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        yield {"type": "image", "filename": filename, "content": image_data}
                        return
                except Exception as e:
                    yield {"error": f"Failed to process image {filename}: {str(e)}"}
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
            
    finally:
        observer.stop()
        observer.join()

# Configure and start the RunPod serverless function
runpod.serverless.start(
    {
        "handler": async_generator_handler,
        "return_aggregate_stream": True,
    }
)