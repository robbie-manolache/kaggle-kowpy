import json
import re
from dataclasses import dataclass
from enum import Enum
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


class Granularity(Enum):
    """Controls the level at which code modifications are tracked"""

    SCRIPT = "script"  # Entire file as one unit
    PARENT = "parent"  # Class/top-level function level
    METHOD = "method"  # Individual method level


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

    def __init__(self, json_text: str, granularity: Granularity = Granularity.METHOD):
        """
        Initialize with JSON text containing files and objects to search for

        Args:
            json_text: JSON string containing "files" and "objects" lists
            granularity: Granularity level for consolidating matches
        """
        # Extract JSON from markdown code block if present
        json_match = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)

        search_data = json.loads(json_text)
        self.search_targets = (
            search_data if isinstance(search_data, list) else []
        )
        self.matches_df: Optional[pd.DataFrame] = None
        self.path_scores: Dict[str, MatchScore] = {}
        self.granularity = granularity

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
                        path = path[len(root_dir_path) :].lstrip("/\\")

                # Calculate path match score
                score = self._calculate_path_score(path, search_path)
                if score.score >= min_path_score:
                    # Check if object name matches
                    if row["name"] == search_object:
                        # If this is a child object, find its parent
                        if row["parent"] is not None:
                            parent_row = df[df["name"] == row["parent"]].iloc[
                                0
                            ]
                            match_row = parent_row
                        else:
                            match_row = row

                        # Store the score for this path
                        self.path_scores[match_row["path"]] = score

                        # Only add if we haven't already added this parent
                        if not any(
                            m.get("node_id") == match_row["node_id"]
                            for m in matches
                        ):
                            matches.append(
                                {
                                    **match_row,
                                    "path_match_score": score.score,
                                    "line_match": (
                                        match_row["start_line"]
                                        <= search_line
                                        <= match_row["end_line"]
                                        if "start_line" in match_row
                                        and "end_line" in match_row
                                        else False
                                    ),
                                }
                            )

        # Create initial matches DataFrame at method level
        matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()
        
        if matches_df.empty:
            self.matches_df = matches_df
            return matches_df
            
        # Consolidate matches based on granularity
        if self.granularity == Granularity.SCRIPT:
            # Group by path and aggregate
            consolidated = matches_df.groupby("path").agg({
                "path_match_score": "max",
                "line_match": "max",
                "start_line": "min",
                "end_line": "max",
                "node_id": "first",  # Keep a reference node_id
                "name": "first",     # Keep first name
                "type": "first",     # Keep first type
                "parent": "first"    # Keep first parent
            }).reset_index()
            
        elif self.granularity == Granularity.PARENT:
            # First get all parent rows
            parent_matches = matches_df[matches_df["parent"].isna()]
            
            # Then get child rows and aggregate them to their parents
            child_matches = matches_df[matches_df["parent"].notna()]
            if not child_matches.empty:
                child_agg = child_matches.groupby("parent").agg({
                    "path": "first",
                    "path_match_score": "max",
                    "line_match": "max"
                })
                
                # Update parent scores where children had better matches
                for parent_name, agg_row in child_agg.iterrows():
                    parent_idx = parent_matches[parent_matches["name"] == parent_name].index
                    if not parent_idx.empty:
                        parent_matches.loc[parent_idx, "path_match_score"] = max(
                            parent_matches.loc[parent_idx, "path_match_score"].iloc[0],
                            agg_row["path_match_score"]
                        )
                        parent_matches.loc[parent_idx, "line_match"] = max(
                            parent_matches.loc[parent_idx, "line_match"].iloc[0],
                            agg_row["line_match"]
                        )
            
            consolidated = parent_matches
            
        else:  # METHOD granularity
            consolidated = matches_df
            
        self.matches_df = consolidated
        return consolidated
