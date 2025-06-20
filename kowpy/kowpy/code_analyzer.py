from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tree_sitter import Tree, Node, Parser
from tree_sitter_languages import get_parser
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
        self.file_extensions: Dict[str, Language] = {
            lang.extension: lang for lang in Language
        }
        self.trees: Dict[str, Tree] = {}

    def get_tree_for(self, file_path: str | Path) -> Optional[Tree]:
        """
        Get the syntax tree for a specific file

        Args:
            file_path: Path to the file whose tree to retrieve

        Returns:
            The syntax tree for the file if it exists, None otherwise
        """
        path_str = str(file_path)
        return self.trees.get(path_str)

    def _get_parser(self, language: str):
        """Get or create parser for a language"""
        if language not in self.parsers:
            self.parsers[language] = get_parser(language)
        return self.parsers[language]

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
        with open(file_path, "rb") as f:
            source_code = f.read()

        parser = self._get_parser(language.value)
        tree = parser.parse(source_code)
        self.trees[str(file_path)] = tree
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

                    # for a class, visit its methods with class name as parent
                    if node.type == "class_definition":
                        for child in node.children:
                            visit_node(child, name)

            # For all nodes (not just those with names), visit children
            # But skip class children as they're handled above
            if node.type != "class_definition":
                for child in node.children:
                    visit_node(child, parent_name)

        visit_node(tree.root_node)
        return objects

    def analyze_directory(self, directory: str | Path) -> pd.DataFrame:
        """
        Analyze all supported files in a directory

        Args:
            directory: Path to the directory to analyze (string or Path object)

        Returns:
            DataFrame containing analyzed code objects with columns:
            path, object_type, name, signature,
            start_line, end_line, parent, node_id
        """
        # Clear existing trees before starting new analysis
        self.trees.clear()
        code_objects: List[CodeObject] = []

        dir_path = Path(directory) if isinstance(directory, str) else directory
        for file_path in dir_path.rglob("*"):
            if file_path.suffix in self.file_extensions:
                language = self.file_extensions[file_path.suffix]
                try:
                    objects = self._analyze_file(file_path, language)
                    code_objects.extend(objects)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")

        df = pd.DataFrame([vars(obj) for obj in code_objects])
        df["node_id"] = df.index
        return df


def analyze_codebase(directory: str) -> pd.DataFrame:
    """
    Analyze a codebase and return a DataFrame of code objects

    Args:
        directory: Path to the codebase directory

    Returns:
        DataFrame with columns: path, object_type, name, signature,
        start_line, end_line, parent, node_id
    """
    analyzer = CodeAnalyzer()
    return analyzer.analyze_directory(directory)
