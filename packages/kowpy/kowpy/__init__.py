"""KowPy - A Python package for code analysis and manipulation"""

__version__ = "0.9.1"

from kowpy.repo_map import (
    map_directory,
    map_directory_json,
)
from kowpy.languages import Language
from kowpy.code_analyzer import (
    CodeAnalyzer,
    analyze_codebase,
)
from kowpy.code_search import CodeSearchMatcher, Granularity
from kowpy.common import CodeSnippet
from kowpy.code_builder import CodeBuilder
from kowpy.prompt import (
    FIXER_PROMPT,
    SEARCH_PROMPT,
    PromptGenerator,
    TextGenerator,
)
from kowpy.pipeline import run_pipeline

__all__ = [
    "map_directory",
    "map_directory_json",
    "Language",
    "CodeAnalyzer",
    "analyze_codebase",
    "CodeSearchMatcher",
    "Granularity",
    "CodeSnippet",
    "CodeBuilder",
    "FIXER_PROMPT",
    "SEARCH_PROMPT",
    "PromptGenerator",
    "TextGenerator",
    "run_pipeline",
]
