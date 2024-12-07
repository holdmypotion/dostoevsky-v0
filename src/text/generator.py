import requests
from typing import Optional, Dict, Any
from loguru import logger


class TextGenerator:
    def __init__(
        self, model_name: str = "llama3.1", host: str = "http://localhost:11434"
    ):
        """
        Initialize the text generator with Ollama

        Args:
            model_name: Name of the Ollama model to use
            host: Ollama API host
        """
        self.model_name = model_name
        self.host = host
        self.generate_endpoint = f"{host}/api/generate"

    def _create_prompt(self, topic: str) -> str:
        """
        Create a structured prompt for poetic text generation
        """
        return f"""Write a deep, contemplative, and poetic paragraph about {topic}.
                  The style should be:
                  - Soothing and calming
                  - Philosophical but accessible
                  - Personal and introspective
                  - Around 4-6 sentences long
                  
                  The tone should be similar to this example:
                  'I love calm people. Those who speak softly but mean every word...'
                  
                  Write the paragraph:"""

    def generate_text(
        self, topic: str, max_length: int = 500, temperature: float = 0.7
    ) -> Optional[str]:
        """
        Generate poetic text using Ollama

        Args:
            topic: The subject to write about
            max_length: Maximum length of generated text
            temperature: Creativity of generation (0.0-1.0)

        Returns:
            Generated text or None if generation fails
        """
        try:
            prompt = self._create_prompt(topic)

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "max_length": max_length,
                    "top_p": 0.9,
                },
            }

            logger.info(f"Generating text for topic: {topic}")
            response = requests.post(self.generate_endpoint, json=payload)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "").strip()

            logger.info(f"Successfully generated text of length: {len(generated_text)}")
            return generated_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating text: {str(e)}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is running and model is loaded
        """
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "healthy",
                    "models": models,
                    "selected_model": self.model_name,
                }
        except requests.exceptions.RequestException:
            pass

        return {"status": "unhealthy", "error": "Cannot connect to Ollama service"}


# Example usage
if __name__ == "__main__":
    generator = TextGenerator()

    # Check if service is healthy
    health = generator.health_check()
    print(f"Service health: {health['status']}")

    if health["status"] == "healthy":
        # Generate sample text
        topic = "the beauty of silence"
        text = generator.generate_text(topic)
        if text:
            print("\nGenerated Text:")
            print(text)
