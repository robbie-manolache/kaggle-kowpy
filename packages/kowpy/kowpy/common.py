from dataclasses import dataclass
from enum import Enum, auto

class SearchMode(Enum):
    """Controls which fields are used for code search matching"""
    LINE_ONLY = auto()
    PARENT_ONLY = auto()
    LINE_AND_PARENT = auto()

JSON_SEARCH_LINE_ONLY = """
```json
[
    {
        "file": "path/to/file1.py",
        "object": "my_function_1",
        "line": 250,
    },
    {
        "file": "path/to/file1.py",
        "object": "my_function_2",
        "line": null,
    },
    {
        "file": "path/to/file2.py",
        "object": "my_function_3",
        "line": 518
    }
]
```

If you cannot find a line number for the object, leave "line" as null.
Pay close attention to error messages for line numbers.
For "file", you may infer relative file paths from import statements.
"""

JSON_SEARCH_PARENT_ONLY = """
```json
[
    {
        "file": "path/to/file1.py",
        "object": "my_function_1",
        "parent": "ClassFoo"
    },
    {
        "file": "path/to/file1.py",
        "object": "my_function_2",
        "parent": null
    },
    {
        "file": "path/to/file2.py",
        "object": "my_function_3",
        "parent": "ClassBar"
    }
]
```

If the object is not part of a Class, leave "parent" as null.
For "file", you may infer relative file paths from import statements.
"""

JSON_SEARCH_LINE_AND_PARENT = """
```json
[
    {
        "file": "path/to/file1.py",
        "object": "my_function_1",
        "line": 250,
        "parent": "ClassFoo"
    },
    {
        "file": "path/to/file1.py",
        "object": "my_function_2",
        "line": null,
        "parent": null
    },
    {
        "file": "path/to/file2.py",
        "object": "my_function_3",
        "line": 518,
        "parent": null
    }
]
```

If you cannot find a line number for the object, leave "line" as null.
Pay close attention to error messages for line numbers.
If the object is not part of a Class, leave "parent" as null.
For "file", you may infer relative file paths from import statements.
"""


@dataclass
class CodeSnippet:
    """Class for code snippets and their metadata for LLM prompts"""

    file_path: str
    node_id: int
    object_name: str
    parent_name: str | None
    code: str | None


def has_substantive_changes(diff_str: str) -> bool:
    """
    Check if a unified diff string contains substantive changes.

    Args:
        diff_str: Unified diff string (from difflib.unified_diff)

    Returns:
        True if diff contains actual code changes, False if only empty lines
    """
    lines = diff_str.splitlines()
    for line in lines:
        # Skip diff headers, context lines, and file marker lines
        skip_header = line.startswith("+++") or line.startswith("---")
        skip_context = not line.startswith("+") and not line.startswith("-")
        if skip_header or skip_context:
            continue
        # Skip empty added/removed lines
        if line in ["+", "-"]:
            continue
        # If we find a non-empty added/removed line, it's substantive
        if line.strip("+- "):
            return True
    return False
