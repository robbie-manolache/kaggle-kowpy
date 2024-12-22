import os
from openai import OpenAI

class AIHelper:
    def __init__(self, api_key=None):
        """Initialize the AI helper with an API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=self.api_key)

    def ask_question(self, message, model="gpt-3.5-turbo"):
        """
        Ask a question and get an AI-generated response.
        
        Args:
            message (str): The question or message to send
            model (str): The OpenAI model to use
            
        Returns:
            str: The AI's response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {str(e)}"
