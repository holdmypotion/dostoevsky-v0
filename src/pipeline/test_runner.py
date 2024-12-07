from src.text.generator import TextGenerator
from src.speech.synthesizer_tts import SpeechSynthesizer

# from src.image.generator import ImageGenerator
from loguru import logger

# Initialize components
text_gen = TextGenerator()
synthesizer = SpeechSynthesizer()
# generator = ImageGenerator()

# Generate text
topic = "the beauty of silence"
generated_text = text_gen.generate_text(topic)
logger.info(f"Generated text: {generated_text}")

if generated_text:
    # Convert to speech
    audio_path = synthesizer.generate_speech(generated_text)
    print(f"Generated audio saved to: {audio_path}")


# Generate an image
# output_files = generator.generate_image(generated_text)

# for output_file in output_files:
#     processed_file = generator.apply_post_processing(output_file)
#     if processed_file:
#         print(f"Processed image: {processed_file}")
