from pathlib import Path
from typing import Optional, Union, Set
import pandas as pd
from .languages import Language
from .code_analyzer import analyze_codebase


class CodeBuilder:
    """Builds code strings from files using DataFrame analysis results"""

    def __init__(
        self,
        analysis_df: Optional[pd.DataFrame] = None,
        directory: Optional[str] = None,
        languages: Optional[Set[Language]] = None,
    ):
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

    def extract_object(self, index: int) -> str:
        """
        Extract code for a specific object from the analysis DataFrame

        Args:
            index: Integer index of the object in the DataFrame

        Returns:
            String containing the extracted code with original indentation
        """
        if self.df is None:
            raise ValueError("No DataFrame has been provided")

        if not 0 <= index < len(self.df):
            raise IndexError(f"Index {index} out of bounds")

        row = self.df.iloc[index]
        return self.extract_code(
            row["path"], row["start_line"], row["end_line"]
        )
