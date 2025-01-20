from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from tree_sitter import Language as TSLanguage, Parser, Tree, Node
from .languages import Language


@dataclass
class CodeObject:
    """Represents a code object found in the analysis"""

    path: str
    object_type: str
    name: str
    signature: str
    start_line: int
    end_line: int
    parent: Optional[str] = None


class CodeAnalyzer:
    """Analyzes code using tree-sitter for multiple languages"""

    def __init__(self):
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, TSLanguage] = {}
        self.file_extensions: Dict[str, Language] = {
            lang.extension: lang for lang in Language
        }

    def add_language(self, language: Language, build_path: str):
        """Add a new language parser"""
        TSLanguage.build_library(build_path, [f"tree-sitter-{language.value}"])
        self.languages[language.value] = TSLanguage(build_path, language.value)
        parser = Parser()
        parser.set_language(self.languages[language.value])
        self.parsers[language.value] = parser

    def _get_signature(self, node: Node, source_code: bytes) -> str:
        """Extract the signature of a code object"""
        return (
            source_code[node.start_byte : node.end_byte]
            .decode("utf-8")
            .split("\n")[0]
        )

    def _get_line_numbers(self, node: Node) -> tuple[int, int]:
        """Get start and end line numbers for a node"""
        return node.start_point[0] + 1, node.end_point[0] + 1

    def _analyze_file(
        self, file_path: Path, language: str
    ) -> List[CodeObject]:
        """Analyze a single file"""
        if language not in self.parsers:
            raise ValueError(f"Parser for language {language} not initialized")

        with open(file_path, "rb") as f:
            source_code = f.read()

        tree = self.parsers[language].parse(source_code)
        objects: List[CodeObject] = []

        def visit_node(node: Node, parent_name: Optional[str] = None):
            if node.type in {
                "function_definition",
                "class_definition",
                "method_definition",
            }:
                name_node = next(
                    (
                        child
                        for child in node.children
                        if child.type == "identifier"
                    ),
                    None,
                )
                if name_node:
                    name = source_code[
                        name_node.start_byte : name_node.end_byte
                    ].decode("utf-8")
                    start_line, end_line = self._get_line_numbers(node)
                    signature = self._get_signature(node, source_code)

                    objects.append(
                        CodeObject(
                            path=str(file_path),
                            object_type=node.type,
                            name=name,
                            signature=signature,
                            start_line=start_line,
                            end_line=end_line,
                            parent=parent_name,
                        )
                    )

                    # If it's a class, visit its methods class name as parent
                    if node.type == "class_definition":
                        for child in node.children:
                            visit_node(child, name)

            # Visit all children
            for child in node.children:
                visit_node(child, parent_name)

        visit_node(tree.root_node)
        return objects

    def analyze_directory(self, directory: Path) -> pd.DataFrame:
        """Analyze all supported files in a directory"""
        code_objects: List[CodeObject] = []

        for file_path in directory.rglob("*"):
            if file_path.suffix in self.file_extensions:
                language = self.file_extensions[file_path.suffix]
                try:
                    objects = self._analyze_file(file_path, language)
                    code_objects.extend(objects)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")

        return pd.DataFrame([vars(obj) for obj in code_objects])


def analyze_codebase(
    directory: str,
    languages: Set[Language],
    build_dir: str = "build/my-languages.so",
) -> pd.DataFrame:
    """
    Analyze a codebase for multiple languages and return a DataFrame

    Args:
        directory: Path to the codebase directory
        languages: Set of Language enum values to analyze
        build_dir: Path where tree-sitter language files will be built

    Returns:
        DataFrame with columns: path, object_type, name, signature,
        start_line, end_line, parent
    """
    analyzer = CodeAnalyzer()

    # Initialize parsers for requested languages
    for lang in languages:
        analyzer.add_language(lang, build_dir)

    return analyzer.analyze_directory(Path(directory))
