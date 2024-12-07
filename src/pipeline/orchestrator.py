from pathlib import Path
from typing import List
from ..config.settings import Settings
from ..text.generator import TextGenerator
from ..speech.synthesizer_chattts import SpeechSynthesizer
from ..image.generator import ImageGenerator
from ..video.compositor import VideoCompositor
from ..utils.logger import logger


class Pipeline:
    def __init__(self, config: Settings):
        self.config = config
        self.text_gen = TextGenerator(config)
        self.speech_syn = SpeechSynthesizer(config)
        self.image_gen = ImageGenerator(config)
        self.compositor = VideoCompositor(config)

    def generate_video(self, prompt: str) -> Path:
        """Execute the full video generation pipeline."""
        try:
            # 1. Generate poetic text
            logger.info("Generating poetic text...")
            text = self.text_gen.generate(prompt)

            # 2. Split into segments (simplified for now)
            segments = text.split("\n\n")

            # 3. Generate speech and images for each segment
            video_segments = []

            for i, segment in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}")

                # Generate speech
                audio_path = self.config.TEMP_DIR / f"segment_{i}_audio.wav"
                self.speech_syn.synthesize(segment, audio_path)

                # Generate image
                image_path = self.config.TEMP_DIR / f"segment_{i}_image.png"
                self.image_gen.generate(segment, image_path)

                # Create video segment
                audio_clip = AudioFileClip(str(audio_path))
                video_segment = self.compositor.create_segment(
                    image_path, audio_path, audio_clip.duration
                )
                video_segments.append(video_segment)

            # 4. Compose final video
            logger.info("Composing final video...")
            final_video = self.compositor.compose_final(video_segments)

            # 5. Save and return result
            output_path = self.config.OUTPUT_DIR / f"poetry_video_{prompt[:30]}.mp4"
            final_video.write_videofile(
                str(output_path), fps=self.config.VIDEO_FPS, codec="libx264"
            )

            logger.info(f"Video generation complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
