from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
    CompositeAudioClip,
)
from loguru import logger


class VideoCompositor:
    def __init__(
        self,
        output_dir: str = "output/videos",
        target_size: Tuple[int, int] = (1080, 1920),  # Instagram reel format
        fps: int = 30,
    ):
        """
        Initialize video compositor

        Args:
            output_dir: Directory for output videos
            target_size: Output video dimensions (width, height)
            fps: Frames per second
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.fps = fps

    def _create_text_overlay(
        self, text: str, duration: float, font_size: int = 40, color: str = "white"
    ) -> TextClip:
        """Create text overlay with fade effects"""
        txt_clip = TextClip(
            text,
            font="SpicyRice-bold",
            color=color,
            size=(self.target_size[0] - 100, None),  # Padding on sides
            method="caption",
            align="center",
            fontsize=font_size,
        )

        txt_clip = txt_clip.set_duration(duration)

        # Add fade in/out
        txt_clip = txt_clip.crossfadein(0.5).crossfadeout(0.5)

        # Center the text
        txt_clip = txt_clip.set_position("center")

        return txt_clip

    def _process_image(self, image_path: str, duration: float) -> ImageClip:
        """Process image with a slight zoom-in effect"""
        img_clip = ImageClip(image_path)

        # Resize image to cover the frame while maintaining aspect ratio
        img_w, img_h = img_clip.size
        target_w, target_h = self.target_size

        scale = max(target_w / img_w, target_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        img_clip = img_clip.resize(new_size)

        # Center crop to target size
        x_center = (new_size[0] - target_w) // 2
        y_center = (new_size[1] - target_h) // 2
        img_clip = img_clip.crop(
            x1=x_center, y1=y_center, x2=x_center + target_w, y2=y_center + target_h
        )

        # Always apply a slight zoom-in effect
        zoom_factor = lambda t: 1 + 0.1 * t
        img_clip = img_clip.resize(zoom_factor)

        # Set duration and position
        img_clip = img_clip.set_duration(duration)
        img_clip = img_clip.set_position("center")

        return img_clip

    def create_video(
        self,
        text_segments: List[str],
        image_paths: List[str],
        audio_path: str,
        background_music_path: Optional[str] = None,
        output_filename: str = "output.mp4",
    ) -> str:
        """
        Create Instagram reel video

        Args:
            text_segments: List of text segments to display
            image_paths: List of image paths to use
            audio_path: Path to main audio file
            background_music_path: Optional background music
            output_filename: Name for output file

        Returns:
            Path to output video file
        """
        try:
            # Load audio and get duration
            main_audio = AudioFileClip(audio_path)
            total_duration = main_audio.duration

            # Calculate segment durations
            segment_duration = total_duration / len(image_paths)

            # Create video clips for each image
            video_clips = []
            for i, (image_path, text) in enumerate(zip(image_paths, text_segments)):
                # Create image clip with zoom
                img_clip = self._process_image(image_path, segment_duration)

                # Create text overlay
                txt_clip = self._create_text_overlay(text, segment_duration)

                # Combine image and text
                composite = CompositeVideoClip([img_clip, txt_clip])
                video_clips.append(composite)

            # Concatenate all clips
            final_video = concatenate_videoclips(video_clips)

            # Add audio
            if background_music_path:
                # Load and loop background music if needed
                bg_music = AudioFileClip(background_music_path)
                if bg_music.duration < total_duration:
                    bg_music = bg_music.loop(duration=total_duration)
                else:
                    bg_music = bg_music.subclip(0, total_duration)

                # Mix audios
                bg_music = bg_music.volumex(0.3)  # Lower background volume
                final_audio = CompositeAudioClip([main_audio, bg_music])
            else:
                final_audio = main_audio

            final_video = final_video.set_audio(final_audio)

            # Write final video
            output_path = self.output_dir / output_filename
            final_video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
            )

            # Clean up
            final_video.close()
            if background_music_path:
                bg_music.close()
            main_audio.close()

            logger.info(f"Successfully created video at {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    compositor = VideoCompositor()

    # Text segments (split your main text into segments)
    text = """In the stillness of silence, I find a quiet refuge from the den of life. It's as if the world itself has taken, just like a breath, holding its gentle sigh, in anticipation of what's to come. In this hushed moment, time stands still, and my mind is free to wander, untethered by the need for words or explanation. The beauty of silence lies not just in its absence of noise, but in its presence of depth. A chance to listen to the whispers of my own heart to feel the gentle stirings of my soul."""

    segments = [
        "In the stillness of silence, I find a quiet refuge from the den of life.",
        "It's as if the world itself has taken, just like a breath, holding its gentle sigh, in anticipation of what's to come.",
        "In this hushed moment, time stands still, and my mind is free to wander, untethered by the need for words or explanation.",
        "The beauty of silence lies not just in its absence of noise, but in its presence of depth.",
        "A chance to listen to the whispers of my own heart to feel the gentle stirings of my soul.",
    ]

    # Create video
    video_path = compositor.create_video(
        text_segments=segments,
        image_paths=[
            "output/images/1.png",
            "output/images/2.png",
            "output/images/3.png",
            "output/images/4.png",
            "output/images/5.png",
        ],
        audio_path="output/audio/silence_poem.wav",
        background_music_path="output/audio/sad_piano.mp3",
        output_filename="silence_reel.mp4",
    )

    print(f"Created video at: {video_path}")
