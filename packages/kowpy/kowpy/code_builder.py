from pathlib import Path
from typing import Optional, Union, Set
import pandas as pd
from difflib import unified_diff
from .languages import Language
from .code_analyzer import analyze_codebase

EXAMPLE = """
Here is your solution

### Snippet 0
```python
def hello_world():
    print("Hello World!")
```

### Snippet 5
```python
potato = 42
if potato == 42:
    print("You're awesome")
```

Hope this `python` code makes you happy.
"""


class CodeBuilder:
    """Builds code strings from files using DataFrame analysis results"""

    def __init__(
        self,
        analysis_df: Optional[pd.DataFrame] = None,
        directory: Optional[str] = None,
        languages: Optional[Set[Language]] = None,
    ):
        self.modified_blocks = {}  # Dictionary to store modified code blocks
        """
        Initialize CodeBuilder with either a DataFrame or directory analysis

        Args:
            analysis_df: DataFrame with columns: path, start_line, end_line
                        (Optional - can be provided later via set_dataframe)
            directory: Path to code directory to analyze (Optional)
            languages: Set of Language enum values to analyze
                        (Required if directory provided)
        """
        if directory is not None:
            if languages is None:
                raise ValueError(
                    "languages must be provided when using directory"
                )
            self.df = analyze_codebase(directory, languages)
        else:
            self.df = analysis_df
        self._validate_df_schema()

    def _validate_df_schema(self) -> None:
        """Validate DataFrame has required columns"""
        if self.df is not None:
            required_cols = {"path", "start_line", "end_line"}
            missing = required_cols - set(self.df.columns)
            if missing:
                raise ValueError(
                    f"DataFrame missing required columns: {missing}"
                )

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set or update the analysis DataFrame

        Args:
            df: DataFrame with columns: path, start_line, end_line
        """
        self.df = df
        self._validate_df_schema()

    def _validate_line_range(
        self, start_line: int, end_line: int, total_lines: int
    ) -> None:
        """Validate line number range"""
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        if end_line < start_line:
            raise ValueError("end_line must be >= start_line")
        if end_line > total_lines:
            raise ValueError(
                f"end_line ({end_line}) exceeds file length ({total_lines})"
            )

    def extract_code(
        self,
        file_path: Union[str, Path],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """
        Extract code from a file, optionally between specific lines

        Args:
            file_path: Path to the source file
            start_line: Starting line number (1-based, optional)
            end_line: Ending line number (1-based, optional)

        Returns:
            String containing the extracted code with original indentation
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line is None or end_line is None:
            return "".join(lines)

        self._validate_line_range(start_line, end_line, len(lines))
        return "".join(lines[start_line - 1 : end_line])

    def extract_object(self, node_id: int) -> str:
        """
        Extract code for a specific object from the analysis DataFrame

        Args:
            node_id: The node_id of the object in the DataFrame

        Returns:
            String containing the extracted code with original indentation
        """
        if self.df is None:
            raise ValueError("No DataFrame has been provided")

        row = self.df[self.df["node_id"] == node_id]
        if row.empty:
            raise ValueError(f"No object found with node_id {node_id}")

        row = row.iloc[0]
        return self.extract_code(
            row["path"], row["start_line"], row["end_line"]
        )

    def _validate_indentation(
        self, original_code: str, modified_code: str
    ) -> bool:
        """
        Validate that modified code maintains proper indentation structure

        Args:
            original_code: Original code string
            modified_code: Modified code string

        Returns:
            True if indentation is valid, False otherwise
        """

        def get_base_indent(code: str) -> int:
            lines = code.splitlines()
            for line in lines:
                if line.strip():
                    return len(line) - len(line.lstrip())
            return 0

        orig_indent = get_base_indent(original_code)
        mod_indent = get_base_indent(modified_code)
        return orig_indent == mod_indent

    def store_modified_block(self, node_id: int, modified_code: str) -> None:
        """
        Store a modified code block for a specific node_id

        Args:
            node_id: The node_id of the object in the DataFrame
            modified_code: Modified code string to store
        """
        if self.df is None:
            raise ValueError("No DataFrame has been provided")

        if node_id not in self.df["node_id"].values:
            raise ValueError(f"No object found with node_id {node_id}")

        original_code = self.extract_object(node_id)
        if not self._validate_indentation(original_code, modified_code):
            raise ValueError(
                "Modified code must maintain original indentation structure"
            )

        self.modified_blocks[node_id] = modified_code

    def compile_file_code(
        self, file_path: Union[str, Path], use_modifications: bool = False
    ) -> str:
        """
        Compile code string for a file, optionally using stored modifications

        Args:
            file_path: Path to the source file
            use_modifications: If True, incorporates stored modifications

        Returns:
            Complete file contents as a string, with any modifications applied
        """
        if self.df is None:
            raise ValueError("No DataFrame has been provided")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not use_modifications:
            return self.extract_code(file_path)

        # Get all rows for this file
        file_rows = self.df[self.df["path"] == str(path)].sort_values(
            "start_line"
        )

        # Adjust any overlapping end_lines
        for i in range(len(file_rows) - 1):
            current_end = file_rows.iloc[i]["end_line"]
            next_start = file_rows.iloc[i + 1]["start_line"]
            if current_end > next_start:
                col_iloc = file_rows.columns.get_loc("end_line")
                file_rows.iloc[i, col_iloc] = next_start - 1

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = []
        current_line = 1

        for idx, (_, row) in enumerate(file_rows.iterrows()):
            # Add any lines before this block
            if current_line < row["start_line"]:
                result.extend(lines[current_line - 1 : row["start_line"] - 1])

            # Add either modified or original block
            if row.node_id in self.modified_blocks:
                result.append(self.modified_blocks[row.node_id])
            else:
                result.extend(lines[row["start_line"] - 1 : row["end_line"]])

            current_line = row["end_line"] + 1

            # If there's a next row, only include lines up to its start
            if idx < len(file_rows) - 1:
                next_start = file_rows.iloc[idx + 1]["start_line"]
                if current_line < next_start:
                    result.extend(lines[current_line - 1 : next_start - 1])
                    current_line = next_start
            # For the last object, include all remaining lines
            elif current_line <= len(lines):
                result.extend(lines[current_line - 1 :])

        return "".join(result)

    def process_snippets(self, text: str) -> list[int]:
        """
        Process a text containing code snippets and store modifications

        The text should contain snippets in the format:
        ### Snippet N
        ```python
        code block
        ```
        where N is the node_id corresponding to the DataFrame

        Args:
            text: Text containing formatted code snippets

        Returns:
            List of node_ids that were successfully processed
        """
        import re
        
        # Pattern to match snippet sections
        pattern = r'### Snippet (\d+)\n```python\n(.*?)```'
        
        processed_ids = []
        
        # Find all matches in the text
        for match in re.finditer(pattern, text, re.DOTALL):
            node_id = int(match.group(1))
            code = match.group(2).rstrip()  # Remove trailing whitespace
            
            try:
                self.store_modified_block(node_id, code)
                processed_ids.append(node_id)
            except ValueError:
                continue  # Skip if node_id doesn't exist or other validation fails
                
        return processed_ids

    def get_modifications_diff(
        self, file_path: Union[str, Path], context_lines: int = 3
    ) -> str:
        """
        Generate unified diff between original and modified versions of a file

        Args:
            file_path: Path to the source file
            context_lines: Number of context lines in diff output (default=3)

        Returns:
            String containing the unified diff
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        original = self.compile_file_code(path, use_modifications=False)
        modified = self.compile_file_code(path, use_modifications=True)

        diff = unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context_lines,
        )

        return "".join(diff)
