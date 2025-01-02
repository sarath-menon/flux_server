import asyncio
import base64
from pathlib import Path
from PIL import Image
import io
from typing import List
import logging
from openai import AsyncOpenAI
import os


PROMPT = """
Write a four sentence caption for this image. In the first sentence describe the style and type (painting, photo, etc) of the image. Describe in the remaining sentences the contents and composition of the image. Only use language that would be used to prompt a text to image model. Do not include usage. Comma separate keywords rather than using "or". Precise composition is important. Avoid phrases like "conveys a sense of" and "capturing the", just use the terms themselves.
""".strip()

class Captioner:
    def __init__(self, max_concurrent: int = 5):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def encode_image(self, image_path: Path) -> str:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode('utf-8')

    async def caption_single_image(
        self,
        image_path: Path,
        caption_path: Path,
        autocaption_prefix: str = "",
        autocaption_suffix: str = ""
    ):
        if caption_path.exists():
            self.logger.info(f"{image_path.name} is already captioned")
            return

        async with self.semaphore:
            try:
                base64_image = self.encode_image(image_path)
                
                modified_prompt = PROMPT
                if autocaption_prefix:
                    modified_prompt += f"\n\nYou must start the caption with '{autocaption_prefix}'."
                if autocaption_suffix:
                    modified_prompt += f"\n\nYou must end the caption with '{autocaption_suffix}'."

                response = await self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": modified_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )

                caption = response.choices[0].message.content
                caption_path.write_text(caption)
                self.logger.info(f"Captioned {image_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")

    def iter_images_captions(self, image_folder: Path):
        for file_path in image_folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
                caption_path = file_path.parent / f"{file_path.stem}.txt"
                yield file_path, caption_path

    def all_images_are_captioned(self, image_folder: Path) -> bool:
        return all(
            caption_path.exists() 
            for _, caption_path in self.iter_images_captions(image_folder)
        )

    async def caption_images(
        self,
        image_folder: Path,
        autocaption_prefix: str = "",
        autocaption_suffix: str = ""
    ):
        self.setup_logging()
        tasks = []
        
        for image_path, caption_path in self.iter_images_captions(image_folder):
            task = self.caption_single_image(
                image_path,
                caption_path,
                autocaption_prefix,
                autocaption_suffix
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
