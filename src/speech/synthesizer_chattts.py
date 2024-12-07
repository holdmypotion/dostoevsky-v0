import torch
import torchaudio
import ChatTTS
from pathlib import Path
from typing import Optional, List, Union
from loguru import logger
import numpy as np


class SpeechSynthesizer:
    def __init__(
        self,
        output_dir: str = "output",
        sample_rate: int = 24000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize speech synthesizer with ChatTTS

        Args:
            output_dir: Directory to save audio files
            sample_rate: Audio sample rate
            device: Device to run inference on
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
        self.device = device
        self.chat = None
        self.current_speaker = None

    def initialize(self):
        """Initialize ChatTTS model"""
        try:
            logger.info("Initializing ChatTTS model...")
            self.chat = ChatTTS.Chat()
            # Set compile=True for better performance if GPU available
            self.chat.load(compile=torch.cuda.is_available())

            # Sample a random speaker for consistency
            self.current_speaker = self.chat.sample_random_speaker()
            logger.info("ChatTTS model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing ChatTTS: {str(e)}")
            return False

    def synthesize_text(
        self,
        text: str,
        output_file: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        add_effects: bool = True,
    ) -> Optional[str]:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_file: Optional output file path
            speaker_embedding: Optional speaker embedding
            add_effects: Whether to add prosodic effects

        Returns:
            Path to generated audio file or None if failed
        """
        try:
            if self.chat is None:
                if not self.initialize():
                    return None

            # Prepare inference parameters
            print(self.current_speaker)
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker_embedding or self.current_speaker,
                temperature=0.3,  # Lower for more stable output
                top_P=0.7,
                top_K=20,
            )

            # Add prosodic effects if requested
            if add_effects:
                params_refine_text = ChatTTS.Chat.RefineTextParams(
                    prompt="[oral_2][laugh_0][break_4]"
                )
            else:
                params_refine_text = None

            # Generate speech
            logger.info(f"Generating speech for text: {text[:50]}...")
            wavs = self.chat.infer(
                text,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )

            # Check if wavs is empty
            if (
                wavs is None
                or len(wavs) == 0
                or (isinstance(wavs, np.ndarray) and wavs.size == 0)
            ):
                logger.error("No audio generated")
                return None

            # Save audio
            output_file = output_file or str(
                self.output_dir / f"speech_{len(text[:10])}_{id}.wav"
            )
            audio_tensor = torch.from_numpy(wavs[0])

            # Handle different torchaudio versions
            try:
                torchaudio.save(
                    output_file, audio_tensor.unsqueeze(0), self.sample_rate
                )
            except:
                torchaudio.save(output_file, audio_tensor, self.sample_rate)

            logger.info(f"Audio saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return None

    def synthesize_batch(
        self, texts: List[str], base_filename: str = "speech"
    ) -> List[Optional[str]]:
        """
        Synthesize multiple texts

        Args:
            texts: List of texts to synthesize
            base_filename: Base name for output files

        Returns:
            List of paths to generated audio files
        """
        output_files = []
        for i, text in enumerate(texts):
            output_file = str(self.output_dir / f"{base_filename}_{i}.wav")
            result = self.synthesize_text(text, output_file)
            output_files.append(result)
        return output_files

    def change_speaker(self) -> None:
        """Sample a new random speaker"""
        if self.chat is None:
            if not self.initialize():
                return None
            self.current_speaker = self.chat.sample_random_speaker()
            logger.info(f"New speaker sampled: {self.current_speaker}")


if __name__ == "__main__":
    synthesizer = SpeechSynthesizer()

    # Test single text
    text = "Hello, this is a test of the speech synthesis system."
    output_file = synthesizer.synthesize_text(text)

    if output_file:
        print(f"Audio generated: {output_file}")

    # Test batch synthesis
    texts = ["This is the first test.", "This is the second test."]
    output_files = synthesizer.synthesize_batch(texts)
    print(f"Batch outputs: {output_files}")
    print(f"Batch outputs: {output_files}")
