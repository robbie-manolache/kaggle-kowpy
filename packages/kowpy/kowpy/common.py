from dataclasses import dataclass


@dataclass
class CodeSnippet:
    """Class for code snippets and their metadata for LLM prompts"""

    file_path: str
    node_id: int
    object_name: str
    parent_name: str | None
    code: str | None
