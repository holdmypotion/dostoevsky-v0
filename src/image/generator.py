import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from multiprocessing import Pool, cpu_count


class ImageGenerator:
    def __init__(
        self,
        output_dir: str = "output/images",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize image generator with Stable Cascade

        Args:
            output_dir: Directory to save generated images
            device: Device to run inference on
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.prior = None
        self.decoder = None

    def initialize(self) -> bool:
        """Initialize Stable Cascade models"""
        try:
            logger.info("Loading Stable Cascade models...")

            # Load prior model
            self.prior = StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            # Load decoder model
            self.decoder = StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Stable Cascade: {str(e)}")
            return False

    def _create_prompt(self, text: str) -> str:
        """
        Create an artistic prompt from input text

        Args:
            text: Base text to create prompt from

        Returns:
            Engineered prompt for image generation
        """
        style_prompt = (
            "oil painting, impressionistic, ethereal, moody, "
            "dark background, dramatic lighting, textured brushstrokes, "
            "artistic, emotional, painterly effect"
        )

        negative_prompt = (
            "text, watermark, signature, frame, border, "
            "sharp details, photorealistic, digital art"
        )

        full_prompt = f"{text}, {style_prompt}"

        return full_prompt, negative_prompt

    def generate_image(
        self,
        text: str,
        output_file: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
    ) -> List[Optional[str]]:
        """
        Generate images from text prompt, splitting text into sentences

        Args:
            text: Text to generate images from
            output_file: Optional base output file path
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier free guidance scale

        Returns:
            List of paths to generated images or None if failed
        """
        try:
            if self.prior is None or self.decoder is None:
                if not self.initialize():
                    return None

            sentences = text.split(".")
            output_files = []

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                prompt, negative_prompt = self._create_prompt(sentence.strip())

                logger.info(f"Generating image for prompt: {prompt[:50]}...")

                prior_output = self.prior(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).image_embeddings

                images = self.decoder(
                    prompt=prompt,
                    image_embeddings=prior_output,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

                sentence_output_file = output_file or str(
                    self.output_dir / f"image_{hash(sentence)}.png"
                )
                images.save(sentence_output_file)
                output_files.append(sentence_output_file)

                logger.info(f"Image saved to {sentence_output_file}")

            return output_files

        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            return None

    def generate_batch(
        self, texts: List[str], base_filename: str = "image"
    ) -> List[Optional[str]]:
        """
        Generate images for multiple texts

        Args:
            texts: List of texts to generate images from
            base_filename: Base name for output files

        Returns:
            List of paths to generated images
        """
        output_files = []
        for i, text in enumerate(texts):
            output_file = str(self.output_dir / f"{base_filename}_{i}.png")
            result = self.generate_image(text, output_file)
            output_files.append(result)
        return output_files

    def apply_post_processing(self, image_path: str) -> Optional[str]:
        """
        Apply post-processing effects to enhance artistic quality

        Args:
            image_path: Path to input image

        Returns:
            Path to processed image
        """
        try:
            image = Image.open(image_path)

            # Add slight blur for painterly effect
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

            # Adjust contrast and saturation
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.9)

            output_path = str(
                Path(image_path).parent / f"processed_{Path(image_path).name}"
            )
            image.save(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return None


if __name__ == "__main__":
    generator = ImageGenerator()

    # Test single image generation
    text = "A solitary figure standing at the edge of darkness, contemplating the void"
    output_file = generator.generate_image(text)

    if output_file:
        print(f"Image generated: {output_file}")

        processed_file = generator.apply_post_processing(output_file)
        if processed_file:
            print(f"Processed image: {processed_file}")
