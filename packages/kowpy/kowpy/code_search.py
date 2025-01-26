import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from .code_analyzer import CodeObject

EXAMPLE = """
```json
[
    {"file": "path/to/file1.py", "object": "my_function_1", "line": 250},
    {"file": "path/to/file1.py", "object": "my_function_2", "line": 212},
    {"file": "path/to/file2.py", "object": "my_function_3", "line": 518}
]
```
"""


@dataclass
class MatchScore:
    """Represents how well a file path matches"""

    path: str
    matched_segments: int
    total_segments: int

    @property
    def score(self) -> float:
        """Calculate match score between 0 and 1"""
        return (
            self.matched_segments / self.total_segments
            if self.total_segments > 0
            else 0.0
        )


class CodeSearchMatcher:
    """Matches code objects from JSON search criteria against analyzed code"""

    def __init__(self, json_text: str):
        """
        Initialize with JSON text containing files and objects to search for

        Args:
            json_text: JSON string containing "files" and "objects" lists
        """
        # Extract JSON from markdown code block if present
        json_match = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)

        search_data = json.loads(json_text)
        self.search_targets = search_data if isinstance(search_data, list) else []
        self.matches_df: Optional[pd.DataFrame] = None
        self.path_scores: Dict[str, MatchScore] = {}

    def _calculate_path_score(
        self, file_path: str, search_path: str
    ) -> MatchScore:
        """Calculate how well a file path matches a search path"""
        file_parts = Path(file_path).parts
        search_parts = Path(search_path).parts

        # Match parts from right to left
        matched = 0
        total = len(file_parts)

        for i in range(min(len(file_parts), total)):
            if file_parts[-(i + 1)] == search_parts[-(i + 1)]:
                matched += 1
            else:
                break

        return MatchScore(
            path=file_path, matched_segments=matched, total_segments=total
        )

    def match_against_df(
        self,
        df: Union[pd.DataFrame, List[CodeObject]],
        min_path_score: float = 0.0,
        directory: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Match search criteria against analyzed code DataFrame

        Args:
            df: DataFrame from CodeAnalyzer
            min_path_score: Minimum path match score (0-1) to include results
            directory: Optional directory to strip from paths before matching

        Returns:
            DataFrame containing only matching rows
        """
        # Convert to DataFrame if List[CodeObject] provided
        if isinstance(df, list):
            df = pd.DataFrame([vars(obj) for obj in df])

        matches = []
        
        for target in self.search_targets:
            search_path = target["file"]
            search_object = target["object"]
            search_line = target["line"]
            
            # Find matching rows for this target
            for _, row in df.iterrows():
                path = str(Path(row["path"]))
                if directory:
                    root_dir_path = str(Path(directory))
                    if path.startswith(root_dir_path):
                        path = path[len(root_dir_path):].lstrip("/\\")
                
                # Calculate path match score
                score = self._calculate_path_score(path, search_path)
                if score.score >= min_path_score:
                    # Check if object name matches
                    if row["name"] == search_object:
                        # Store the score for this path
                        self.path_scores[row["path"]] = score
                        matches.append({
                            **row,
                            "path_match_score": score.score,
                            "line_match": (
                                row["start_line"] <= search_line <= row["end_line"]
                                if "start_line" in row and "end_line" in row
                                else False
                            )
                        })

        self.matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()

        return self.matches_df
