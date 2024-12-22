import os
from enum import Enum
from typing import Optional, Union, List
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class AIHelper:
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, api_key: Optional[str] = None):
        """
        Initialize the AI helper with a provider and API key.
        
        Args:
            provider: The AI provider to use (OpenAI or Anthropic)
            api_key: Optional API key. If not provided, will look for environment variables
        """
        load_dotenv()
        self.provider = provider
        
        if provider == AIProvider.OPENAI:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")
            self.client = Anthropic(api_key=self.api_key)

    def construct_prompt(self, 
                        question: str, 
                        reference_text: Optional[str] = None,
                        initial_response: Optional[str] = None) -> str:
        """
        Construct a prompt combining reference text, question and optional initial response.
        
        Args:
            question: The user's question
            reference_text: Optional reference text to provide context
            initial_response: Optional initial response to build upon
            
        Returns:
            The constructed prompt
        """
        prompt_parts = []
        
        if reference_text:
            prompt_parts.append(f"Reference text:\n{reference_text}\n")
        
        prompt_parts.append(f"Question: {question}")
        
        if initial_response:
            prompt_parts.append(f"\nInitial response: {initial_response}")
            prompt_parts.append("\nPlease improve or correct the above response if needed.")
            
        return "\n".join(prompt_parts)

    def ask_question(self,
                    message: str,
                    system_prompt: Optional[str] = None,
                    model: Optional[str] = None) -> str:
        """
        Ask a question and get an AI-generated response.
        
        Args:
            message: The question or message to send
            system_prompt: Optional system prompt to guide the AI's behavior
            model: The model to use (defaults to gpt-3.5-turbo for OpenAI or claude-3-opus for Anthropic)
            
        Returns:
            The AI's response
        """
        try:
            if self.provider == AIProvider.OPENAI:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})
                
                response = self.client.chat.completions.create(
                    model=model or "gpt-3.5-turbo",
                    messages=messages
                )
                return response.choices[0].message.content
            else:
                message_content = f"{system_prompt}\n\n{message}" if system_prompt else message
                response = self.client.messages.create(
                    model=model or "claude-3-opus",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": message_content}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            return f"Error getting response: {str(e)}"
