from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict, Union, Callable
from .common import CodeSnippet
import numpy as np


@dataclass
class PromptGenerator:
    """A class to generate system and user prompts for LLM input."""

    system_prompt: Union[str, Callable[..., str]]
    user_prompt: Union[str, Callable[..., str]]

    def execute_prompt(
        self, prompt: Union[str, Callable[..., str]], **kwargs
    ) -> str:
        """Execute prompt if it's a Callable, or return if it's a string."""
        if isinstance(prompt, str):
            return prompt
        return prompt(**kwargs)

    def get_system_prompt(self, **kwargs) -> str:
        """Execute the system prompt with given kwargs."""
        return self.execute_prompt(self.system_prompt, **kwargs)

    def get_user_prompt(self, **kwargs) -> str:
        """Execute the user prompt with given kwargs."""
        return self.execute_prompt(self.user_prompt, **kwargs)

    def generate_messages(
        self,
        system_kwargs: dict = None,
        user_kwargs: dict = None,
        verbose: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Generate messages list for LLM input with separate kwargs for each
        prompt type.

        Args:
            system_kwargs: Keywords arguments for system prompt
            user_kwargs: Keyword arguments for user prompt
            verbose: If True, log the generated messages

        Returns:
            List of message dictionaries with role and content
        """
        system_kwargs = system_kwargs or {}
        user_kwargs = user_kwargs or {}

        system_text = self.get_system_prompt(**system_kwargs)
        user_text = self.get_user_prompt(**user_kwargs)

        if verbose:
            print("\n>>> SYSTEM PROMPT START <<<\n")
            print(system_text)
            print("\n>>> SYSTEM PROMPT END <<<")
            print("\n>>> USER PROMPT START <<<\n")
            print(user_text)
            print("\n>>> USER PROMPT END <<<")

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]


def search_user_prompt(problem: str) -> str:
    """Generate prompt to search for objects/files in a repo"""

    return (
        f"""
You are working in a code repo on the following problem:

{problem}

Which files and objects require modification to resolve the issue?
"""
        + """
Your response must be in the following format:
```json
[
    {"file": "path/to/file1.py", "object": "my_function_1", "line": 250},
    {"file": "path/to/file1.py", "object": "my_function_2", "line": 212},
    {"file": "path/to/file2.py", "object": "my_function_3", "line": 518}
]
```
"""
        + f"""
Do not attempt to solve the issue.
Make sure to consider all relevant file paths and objects, \
especially those shown in error messages related to the repo.
"""
    )


SEARCH_PROMPT = PromptGenerator(
    system_prompt="""
You are a software engineering assistant resolving issues in a Python code \
repository. You are responsible only for identifying the relevant files and \
objects that will be required by someone else to solve the issue.
""",
    user_prompt=search_user_prompt,
)


def fixer_user_prompt(
    problem: str,
    snippets: List[CodeSnippet],
) -> str:

    fix_prompt = f"""
You are working in a code repo on the following problem:

{problem}

Do not fix any of the code shown above.
Only fix the snippets shown below in a way that fixes the problem:
"""

    snippet_ids = []
    for snip in snippets:
        snippet_ids.append(snip.node_id)
        body = f"\n### Snippet {snip.node_id}"
        obj, parent = snip.object_name, snip.parent_name
        if parent:
            desc = f"method `{obj}` from `{parent}` class"
        else:
            desc = f"function `{obj}`"
        body += f"\nContains the code for {desc}:"
        body += "\n```python\n\n" + snip.code + "\n```\n\n"
        fix_prompt += body

    example_ids = [15, 42]
    while any([i in snippet_ids for i in example_ids]):
        example_ids = np.random.randint(low=0, high=500, size=2)
    snippet_ids = ", ".join([str(i) for i in snippet_ids])

    fix_prompt += f"""
Revise only the labeled snippets and return the updated code.
Use the following example format for the response:

### Snippet {example_ids[0]}
```python
def hello_world():
    print("Hello World!")
```

### Snippet {example_ids[1]}
```python
def goodbye_cruel_world():
    print("Goodbye Cruel World!")
```

This is an example only. In your respose you must only provide fixes for \
the snippets with IDs {snippet_ids}.

Maintain the original indentation from the snippets.
Make sure to label each response snippet correctly.

If you are confident the code changes have resolved \
the issue, then end your response with:

```json
{"status": "SUCCESS"}
```

If more work is required to resolve the issue \
then end your response with:

```json
{"status": "INCOMPLETE"}
```

"""

    return fix_prompt


FIXER_PROMPT = PromptGenerator(
    system_prompt="""
You are a software engineering assistant resolving issues in a Python code \
repository. You are responsible for patching code that addresses problems \
reported by users.
""",
    user_prompt=fixer_user_prompt,
)


class TextGenerator:
    """A class to handle text generation using transformer models."""

    def __init__(self, model_name: str, max_tokens: int = 4096):
        """
        Initialize the text generator with a model.

        Args:
            model_name (str): Path or name of the model to load
            max_tokens (int): Maximum number of input tokens allowed
                (default: 4096)
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

    def generate(self, max_new_tokens: int | None = None) -> None:
        """
        Generate text based on the prepared input.

        Args:
            max_new_tokens (int): Maximum number of tokens to generate
        """
        if self.model_inputs is None:
            raise ValueError("Input not prepared. Call prepare_input first.")

        if self.prompt_tokens_over_limit:
            raise ValueError(
                f"Input length ({self.input_length} tokens) exceeds maximum "
                f"allowed tokens ({self.max_tokens})"
            )

        generated_ids = self.model.generate(
            **self.model_inputs,
            max_new_tokens=max_new_tokens or self.max_tokens,
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

    @staticmethod
    def parse_status(response_text: str) -> bool:
        """
        Parse the response text to determine if status is SUCCESS.
        
        Args:
            response_text (str): The response text to parse
            
        Returns:
            bool: True if status is SUCCESS, False otherwise
        """
        try:
            # Look for any JSON-like strings in the response
            start_pos = response_text.find('{')
            while start_pos != -1:
                end_pos = response_text.find('}', start_pos) + 1
                if end_pos == 0:  # No closing brace found
                    break
                    
                json_str = response_text[start_pos:end_pos]
                try:
                    status_obj = json.loads(json_str)
                    if "status" in status_obj:
                        return status_obj["status"] == "SUCCESS"
                except json.JSONDecodeError:
                    pass
                
                # Look for next JSON object
                start_pos = response_text.find('{', end_pos)
                
            return False  # No valid status found
            
        except Exception:
            return False  # Any parsing error defaults to False
