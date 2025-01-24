import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from .code_analyzer import CodeObject


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
        self.search_files = search_data.get("files", [])
        self.search_objects = search_data.get("objects", [])

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
        root_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Match search criteria against analyzed code DataFrame

        Args:
            df: DataFrame from CodeAnalyzer
            min_path_score: Minimum path match score (0-1) to include results
            root_dir: Optional root directory to strip from paths before matching

        Returns:
            DataFrame containing only matching rows
        """
        # Convert to DataFrame if List[CodeObject] provided
        if isinstance(df, list):
            df = pd.DataFrame([vars(obj) for obj in df])

        # Calculate path scores for all unique paths
        unique_paths = df["path"].unique()
        for path in unique_paths:
            # Strip root_dir from path if provided
            match_path = str(Path(path))
            if root_dir:
                root_dir_path = str(Path(root_dir))
                if match_path.startswith(root_dir_path):
                    match_path = match_path[len(root_dir_path):].lstrip('/\\')
            best_score = 0
            best_match = None

            # Find best matching search path
            for search_path in self.search_files:
                score = self._calculate_path_score(match_path, search_path)
                if score.score > best_score:
                    best_score = score.score
                    best_match = score

            if best_match and best_score >= min_path_score:
                self.path_scores[path] = best_match

        # Filter DataFrame to matching paths and objects
        path_mask = df["path"].isin(self.path_scores.keys())
        object_mask = df["name"].isin(self.search_objects)

        self.matches_df = df[path_mask & object_mask].copy()

        # Add score column
        self.matches_df["path_match_score"] = self.matches_df["path"].map(
            lambda x: self.path_scores[x].score
        )

        return self.matches_df
