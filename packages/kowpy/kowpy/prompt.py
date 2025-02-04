from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class TextGenerator:
    """A class to handle text generation using transformer models."""

    def __init__(self, model_name: str):
        """
        Initialize the text generator with a model.

        Args:
            model_name (str): Path or name of the model to load
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.messages = None
        self.text = None
        self.model_inputs = None
        self.generated_ids = None
        self.response = None

    def _validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate the format of input messages."""
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")

        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must have 'role' and 'content' keys"
                )
        return True

    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the messages for generation.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
        """
        if self._validate_messages(messages):
            self.messages = messages

    def prepare_input(self) -> None:
        """Prepare the input for the model."""
        if self.messages is None:
            raise ValueError("Messages not set. Call set_messages first.")

        self.text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.model_inputs = self.tokenizer(
            [self.text], return_tensors="pt"
        ).to(self.model.device)

    def generate(self, max_new_tokens: int = 512) -> None:
        """
        Generate text based on the prepared input.

        Args:
            max_new_tokens (int): Maximum number of tokens to generate
        """
        if self.model_inputs is None:
            raise ValueError("Input not prepared. Call prepare_input first.")

        generated_ids = self.model.generate(
            **self.model_inputs,
            max_new_tokens=max_new_tokens,
        )
        self.generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                self.model_inputs.input_ids, generated_ids
            )
        ]

        self.response = self.tokenizer.batch_decode(
            self.generated_ids, skip_special_tokens=True
        )

    def get_response(self, index: int = 0) -> str:
        """
        Get the generated response at the specified index.

        Args:
            index (int): Index of the response to return (default: 0)

        Returns:
            str: Generated response
        """
        if self.response is None:
            raise ValueError("No response generated. Call generate first.")
        return self.response[index]
