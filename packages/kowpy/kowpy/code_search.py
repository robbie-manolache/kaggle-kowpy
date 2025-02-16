import json
import operator
import re
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from .code_analyzer import CodeObject
from .common import CodeSnippet, SearchMode


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

    def __init__(
        self,
        json_text: str,
        search_mode: SearchMode,
        granularity: Granularity = Granularity.METHOD,
    ):
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
        # Add target_id to each search target
        self.search_targets = []
        self.search_mode = search_mode

        for target in search_data if isinstance(search_data, list) else []:
            # Handle methods field if present
            methods = target.pop("methods", None)

            if methods is not None:
                # If methods list is not empty, create entries for each method
                if methods:
                    line = target.get("line")
                    for method in methods:
                        method_target = {
                            "file": target["file"],
                            "object": method,
                            "line": line,
                            "parent": target["object"],
                            "target_id": len(self.search_targets),
                        }
                        self.search_targets.append(method_target)
                else:
                    # Empty methods list, just add null parent
                    target["parent"] = None
                    target["target_id"] = len(self.search_targets)
                    self.search_targets.append(target)
            else:
                # No methods field, handle traditional format
                if search_mode == SearchMode.LINE_ONLY:
                    target.setdefault("parent", None)
                elif search_mode == SearchMode.PARENT_ONLY:
                    target.setdefault("line", None)

                target["target_id"] = len(self.search_targets)
                self.search_targets.append(target)

        self.matches_df: Optional[pd.DataFrame] = None
        self.ranked_matches_df: Optional[pd.DataFrame] = None
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

        for i in range(min(len(file_parts), len(search_parts))):
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
            search_line = target.get("line") or -1
            search_parent = target.get("parent")

            target_id = target["target_id"]
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
                        # Store the score for this path
                        self.path_scores[row["path"]] = score

                        # Check parent match
                        code_parent = row.get("parent")
                        if code_parent and search_parent:
                            parent_match = search_parent == code_parent
                        else:
                            parent_match = False

                        matches.append(
                            {
                                **row,
                                "target_id": target_id,
                                "path_match_score": score.score,
                                "parent_match": parent_match,
                                "line_match": (
                                    row["start_line"]
                                    <= search_line
                                    <= row["end_line"]
                                    if "start_line" in row
                                    and "end_line" in row
                                    and search_line > 0
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
            agg_dict = {
                "path_match_score": "max",
                "parent_match": "max",
                "line_match": "max",
                "start_line": "min",
                "end_line": "max",
            }
            consolidated = (
                matches_df.groupby("path").agg(agg_dict).reset_index()
            )

        elif self.granularity == Granularity.PARENT:
            # Get direct parent matches
            parent_matches = matches_df[matches_df["parent"].isna()]

            # Get child matches and aggregate to parent level
            child_matches = matches_df[matches_df["parent"].notna()]
            if not child_matches.empty:
                # Dedupe best child matches by parent
                key_cols = ["parent", "path"]
                sort_cols = ["path_match_score", "parent_match", "line_match"]
                child_cols = [*key_cols, "target_id", *sort_cols]
                child_agg = child_matches[child_cols].sort_values(
                    sort_cols, ascending=[False, False, False]
                )
                child_agg = child_agg.drop_duplicates(subset=key_cols)

                # Rename column to match original df
                child_agg = child_agg.rename(columns={"parent": "name"})

                # Get parent info from original df for child matches
                parent_info = df[df["parent"].isna()].copy()
                child_parents = child_agg.merge(
                    parent_info, on=["name", "path"]
                )

                # Combine direct parent matches with children's parent matches
                if not parent_matches.empty:
                    consolidated = pd.concat([parent_matches, child_parents])
                    # If a parent exists in both, keep one with higher score
                    consolidated = consolidated.sort_values(
                        ["name", "path_match_score", "line_match"],
                        ascending=[True, False, False],
                    ).drop_duplicates(subset=["name"], keep="first")
                else:
                    consolidated = child_parents
            else:
                consolidated = parent_matches

        else:  # METHOD granularity
            consolidated = matches_df

        self.matches_df = consolidated
        return consolidated

    def rank_matches(
        self,
        deduplicate: bool = True,
        drop_zero_score: bool = False,
    ) -> pd.DataFrame:
        """
        Rank matches by path_match_score and line_match to find best matches.

        Args:
            deduplicate: If True, removes parent entries when child methods
                exist in the same file (only applies to METHOD granularity)
            drop_zero_score: If True, removes matches where path_match_score,
                line_match and parent_match are all 0.

        Returns:
            DataFrame containing only the best match for each search target
        """
        if self.matches_df is None or self.matches_df.empty:
            self.ranked_matches_df = pd.DataFrame()
            return self.ranked_matches_df

        # Determine sort columns based on search mode
        sort_cols = ["target_id", "path_match_score"]
        if self.search_mode == SearchMode.LINE_ONLY:
            sort_cols.append("line_match")
        elif self.search_mode == SearchMode.PARENT_ONLY:
            sort_cols.append("parent_match")
        elif self.search_mode == SearchMode.LINE_AND_PARENT:
            sort_cols.extend(["line_match", "parent_match"])

        # Sort matches using appropriate columns
        sorted_df = self.matches_df.sort_values(
            by=sort_cols,
            ascending=[True, False] + [False] * (len(sort_cols) - 2),
        )

        # Keep only the best match for each target_id
        best_matches = sorted_df.groupby("target_id").first().reset_index()

        # Remove matches with no meaningful match criteria
        if drop_zero_score:
            valid_match = [
                best_matches["path_match_score"] > 0,
                best_matches["line_match"],
                best_matches["parent_match"],
            ]
            best_matches = best_matches[reduce(operator.or_, valid_match)]

        # For METHOD granularity, remove parent entries if with child matches
        if self.granularity == Granularity.METHOD and deduplicate:
            # Get all parent names and their files that have child entries
            parents_with_children = best_matches[
                best_matches["parent"].notna()
            ][["parent", "path"]]

            # Remove parent entries that have children in the same file
            best_matches = best_matches[
                ~(
                    (best_matches["parent"].isna())  # Is a parent entry
                    & best_matches.apply(  # Has children in same file
                        lambda x: (
                            (parents_with_children["parent"] == x["name"])
                            & (parents_with_children["path"] == x["path"])
                        ).any(),
                        axis=1,
                    )
                )
            ]

        # keep unique node ID only
        best_matches = best_matches.drop_duplicates(subset=["node_id"])

        self.ranked_matches_df = best_matches
        return self.ranked_matches_df

    def get_ranked_snippets(self) -> List[CodeSnippet]:
        """
        Convert ranked matches DataFrame to list of CodeSnippet objects.

        Returns:
            List of CodeSnippet objects for each ranked match
        """
        if self.ranked_matches_df is None or self.ranked_matches_df.empty:
            return []

        snippets = []
        for _, row in self.ranked_matches_df.iterrows():
            snippet = CodeSnippet(
                file_path=row["path"],
                node_id=row["node_id"],
                object_name=row["name"],
                parent_name=row.get("parent"),  # NOTE: parent may be None
                code=None,
            )
            snippets.append(snippet)

        return snippets
