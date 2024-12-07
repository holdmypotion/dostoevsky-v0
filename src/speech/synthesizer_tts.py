import os
from pathlib import Path
from typing import Optional
import torch
from TTS.api import TTS
from loguru import logger


class SpeechSynthesizer:
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/glow-tts",
        output_dir: str = "output/audio",
        device: Optional[str] = None,
    ):
        """
        Initialize the speech synthesizer

        Args:
            model_name: Name of the TTS model to use
            output_dir: Directory to save audio files
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing TTS model on {self.device}")

        try:
            self.tts = TTS(model_name)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise

    def generate_speech(
        self,
        text: str,
        speaker_wav: str = None,
        language: str = "en",
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Convert text to speech

        Args:
            text: Text to convert to speech
            speaker_wav: Path to speaker reference audio for voice cloning
            language: Language code for multilingual models
            output_filename: Optional filename for the output audio

        Returns:
            Path to generated audio file
        """
        try:
            # Generate output filename if not provided
            if output_filename is None:
                output_filename = f"speech_{hash(text)}.wav"

            output_path = self.output_dir / output_filename

            logger.info(f"Generating speech for text: {text[:50]}...")

            # Generate speech
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                # language=language,
                file_path=str(output_path),
            )

            logger.info(f"Successfully generated speech at {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate speech: {str(e)}")
            raise

    def generate_speech_segments(
        self, segments: list[str], speaker_wav: str = None, language: str = "en"
    ) -> list[str]:
        """
        Convert multiple text segments to speech

        Args:
            segments: List of text segments
            speaker_wav: Path to speaker reference audio
            language: Language code

        Returns:
            List of paths to generated audio files
        """
        audio_paths = []

        for i, text in enumerate(segments):
            output_filename = f"segment_{i}.wav"
            audio_path = self.generate_speech(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                output_filename=output_filename,
            )
            audio_paths.append(audio_path)

        return audio_paths

    def cleanup_old_files(self, max_files: int = 100):
        """Delete old audio files if too many accumulate"""
        files = list(self.output_dir.glob("*.wav"))
        if len(files) > max_files:
            # Sort by creation time and delete oldest
            files.sort(key=lambda x: x.stat().st_ctime)
            for file in files[:-max_files]:
                file.unlink()
                logger.info(f"Deleted old audio file: {file}")


if __name__ == "__main__":
    synthesizer = SpeechSynthesizer()

    # Single text to speech
    audio_path = synthesizer.generate_speech(
        text="Hello, this is a test of the speech synthesis system.", language="en"
    )
    print(f"Generated audio saved to: {audio_path}")

    # Multiple segments
    segments = [
        "This is the first segment.",
        "Here is the second segment.",
        "And finally, the third segment.",
    ]
    audio_paths = synthesizer.generate_speech_segments(segments)
    print(f"Generated {len(audio_paths)} audio segments")
