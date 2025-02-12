from dataclasses import dataclass

JSON_OUTPUT_EXAMPLE = """
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
