from enum import Enum, auto
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json

MAX_TOKENS = 3072


class TextGenerator:
    """A class to handle text generation using transformer models."""

    def __init__(self, model_name: str, max_tokens: int = MAX_TOKENS):
        """
        Initialize the text generator with a model.

        Args:
            model_name (str): Path or name of the model to load
            max_tokens (int): Maximum number of input tokens allowed
                (default: MAX_TOKENS)
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.messages = None
        self.text = None
        self.model_inputs = None
        self.input_length = None
        self.prompt_tokens_over_limit = False
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

    def reset_max_tokens(self, max_tokens: int) -> None:
        """
        Set the maximum number of tokens allowed.

        Args:
            max_tokens (int): Maximum number of input tokens allowed
        """
        self.max_tokens = max_tokens
        self.prompt_tokens_over_limit = False

    def prepare_input(self) -> None:
        """
        Prepare the input for the model.

        Raises:
            ValueError: If messages not set or input exceeds token limit
        """
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

        # Check token length
        self.input_length = self.model_inputs.input_ids.shape[1]
        if self.input_length > self.max_tokens:
            import warnings

            warnings.warn(
                f"Input length ({self.input_length} tokens) exceeds "
                f"maximum allowed tokens ({self.max_tokens})"
            )
            self.prompt_tokens_over_limit = True

    def generate(self, **kwargs) -> None:
        """
        Generate text based on the prepared input.

        Args:
            **kwargs: Keyword arguments to pass to model.generate().
                Defaults to max_new_tokens=self.max_tokens if not provided.
        """
        if self.model_inputs is None:
            raise ValueError("Input not prepared. Call prepare_input first.")

        if self.prompt_tokens_over_limit:
            raise ValueError(
                f"Input length ({self.input_length} tokens) exceeds maximum "
                f"allowed tokens ({self.max_tokens})"
            )

        # Set default max_new_tokens if not in kwargs
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = self.max_tokens * 1.25

        generated_ids = self.model.generate(**self.model_inputs, **kwargs)
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

    class ResponseStatus(Enum):
        """Enum for response status parsing results"""

        SUCCESS = auto()
        INCOMPLETE = auto()
        UNKNOWN = auto()

    @staticmethod
    def parse_status(response_text: str) -> "TextGenerator.ResponseStatus":
        """
        Parse the response text to determine the completion status.
        Looks specifically for ```json blocks containing status.

        Args:
            response_text (str): The response text to parse

        Returns:
            ResponseStatus: SUCCESS, INCOMPLETE, or UNKNOWN enum status
        """
        try:
            # Look for ```json blocks
            json_blocks = response_text.split("```json")
            for block in json_blocks[1:]:  # Skip first split before any json
                # Find the end of the block
                end_pos = block.find("```")
                if end_pos == -1:
                    continue

                # Extract and parse the JSON content
                json_str = block[:end_pos].strip()
                try:
                    status_obj = json.loads(json_str)
                    if "status" in status_obj:
                        status = status_obj["status"].upper()
                        if status == "SUCCESS":
                            return TextGenerator.ResponseStatus.SUCCESS
                        elif status == "INCOMPLETE":
                            return TextGenerator.ResponseStatus.INCOMPLETE
                except json.JSONDecodeError:
                    continue

            # No valid status found
            return TextGenerator.ResponseStatus.UNKNOWN

        except Exception:
            # Any parsing error defaults to UNKNOWN
            return TextGenerator.ResponseStatus.UNKNOWN
