from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Union, Callable
from .common import (
    SearchMode,
    JSON_SEARCH_LINE_ONLY,
    JSON_SEARCH_PARENT_ONLY,
    JSON_SEARCH_LINE_AND_PARENT,
    JSON_SEARCH_LINE_METHODS,
    CodeSnippet,
)
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


def search_user_prompt_v0(problem: str, search_mode: SearchMode) -> str:
    """
    Generate prompt to search for objects/files in a repo

    Args:
        problem: Problem statement to analyze
        search_mode: Controls which JSON template to use
    """
    json_template = {
        SearchMode.LINE_ONLY: JSON_SEARCH_LINE_ONLY,
        SearchMode.PARENT_ONLY: JSON_SEARCH_PARENT_ONLY,
        SearchMode.LINE_AND_PARENT: JSON_SEARCH_LINE_AND_PARENT,
    }[search_mode]

    return (
        f"""
You are working in a code repo on the following problem statement:

{problem}

Which files and objects require modification to resolve the issue?
Your response must be in the following format:
"""
        + json_template
        + f"""
Do not attempt to solve the issue.
Make sure to consider all relevant file paths and objects, \
especially those shown in error messages related to the repo.
"""
    )


def search_user_prompt_with_example(problem: str) -> str:
    """
    Generate prompt to search for objects/files in a repo

    Args:
        problem: Problem statement to analyze
    """
    return (
        """
I want you to inspect sample code and error messages to identify Python \
objects that require modification to solve an issue. Your response must be in \
JSON format with the following fields: "file", "object", "line", "parent".
Below are some examples:

### EXAMPLE 1

#### USER INPUT
I tried running the following code:
```
from module import foo

Test = foo()
foo.some_method(x=42)
```
This raised the following error:
```
Traceback (most recent call last):
  File "/path/to/main.py", line 4, in <module>
    Test.some_method()
  File "/path/to/class_file.py", line 42, in some_method
    raise ValueError("An error occurred")
```

#### YOUR RESPONSE
```json
[
    {
        "file": "path/to/class_file.py",
        "object": "foo",
        "line": null,
        "parent": null
    },
    {
        "file": "path/to/class_file.py",
        "object": "some_method",
        "line": 42,
        "parent": "foo"
    },
]
```

### EXAMPLE 2

#### USER INPUT
I tried running the following code:
```
from package.module import hello_world

hello_world()
```
I got an unexpected result: `"Hello Mars!"`

#### YOUR RESPONSE
```json
[
    {
        "file": "package/module.py",
        "object": "hello_world",
        "line": null,
        "parent": null
    },
]
```

Can you provide a response in the same format for the following user input:

"""
        + problem
    )


def search_user_prompt_detailed(problem: str) -> str:
    """
    Generate prompt to search for objects/files in a repo

    Args:
        problem: Problem statement to analyze
    """
    return f"""
I want you to inspect a GitHub issue and identify Python objects from \
code samples and/or error tracebacks provided by users. For each object \
you should identify the following fields:

- "object": the name of the Python object. In error tracebacks, this is \
typically preceded by `in`
- "file": path to the file, usually visible in error tracebacks. For code \
samples, the path might need to be inferred from import statements
- "line": line number, usually only visible in error traceback messages. \
Return null if not available
- "methods": if the object is a class, list any relevant methods else \
return an empty list.

Your response output must follow the JSON format shown in this example:

{JSON_SEARCH_LINE_METHODS}

All of the values above are made up. You must identify values from \
within the GitHub issue below only.

Below is the GitHub issue you must focus on, carefully inspecting the \
sample code and all error traceback messages, returning a response in \
the JSON format shown above:

{problem}
"""


class SearchPromptType(Enum):
    V0 = auto()
    EG = auto()
    DT = auto()


def search_user_prompt(
    problem: str, prompt_type: SearchPromptType, **kwargs
) -> str:
    """
    Generate prompt to search for objects/files in a repo

    Args:
        problem: Problem statement to analyze
        prompt_type: SearchPromptType to determine which prompt generator
            function gets called
        kwargs: Kwargs to pass to the prompt generator function
    """
    if prompt_type == SearchPromptType.V0:
        return search_user_prompt_v0(problem, kwargs.get("search_mode"))
    elif prompt_type == SearchPromptType.EG:
        return search_user_prompt_with_example(problem)
    elif prompt_type == SearchPromptType.DT:
        return search_user_prompt_detailed(problem)
    else:
        raise ValueError(f"Unexpected prompt type {prompt_type}")


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
"""

    fix_prompt += """
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
