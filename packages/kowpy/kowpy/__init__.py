"""KowPy - A Python package for code analysis and manipulation"""

__version__ = "0.11.0"

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
from kowpy.common import CodeSnippet, SearchMode, has_substantive_changes
from kowpy.code_builder import CodeBuilder
from kowpy.prompt import (
    FIXER_PROMPT,
    SEARCH_PROMPT,
    PromptGenerator,
    SearchPromptType,
)
from kowpy.model import TextGenerator
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
    "has_substantive_changes",
    "CodeBuilder",
    "FIXER_PROMPT",
    "SEARCH_PROMPT",
    "SearchMode",
    "PromptGenerator",
    "SearchPromptType",
    "TextGenerator",
    "run_pipeline",
]
