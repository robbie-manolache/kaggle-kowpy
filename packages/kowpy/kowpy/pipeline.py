from pathlib import Path
import pandas as pd

from .code_analyzer import analyze_codebase
from .code_builder import CodeBuilder
from .code_search import Granularity, CodeSearchMatcher
from .prompt import (
    FIXER_PROMPT,
    SEARCH_PROMPT,
    TextGenerator,
)


def run_pipeline(
    repo_path: str,
    problem: str,
    model_name: str,
) -> str | None:
    """
    Run the complete analysis and modification pipeline on a repository.

    This function:
    1. Extracts problem details from the input DataFrame
    2. Generates a search query using an LLM
    3. Analyzes the codebase
    4. Matches relevant code sections
    5. Generates fixes using an LLM
    6. Returns a unified diff of proposed changes

    Args:
        df: DataFrame containing repository and issue information
        root_dir: Root directory containing repository data
        issue_index: Index of the issue to process in the DataFrame
        model_name: Name of the LLM model to use

    Returns:
        String containing unified diff of proposed changes, or None if it fails
    """

    # Prepare common kwargs for prompts
    base_kwargs = {"problem": problem}

    # Initialize text generator with model
    # Generate search criteria using LLM
    search_msg = SEARCH_PROMPT.generate_messages(user_kwargs=base_kwargs)
    txtgen = TextGenerator(model_name)

    try:
        txtgen.set_messages(search_msg)
        txtgen.prepare_input()
        txtgen.generate()
        search_output = txtgen.get_response()
    except Exception:
        return None
    
    # Analyze codebase and find relevant code sections
    df_code = analyze_codebase(directory=repo_path)
    csm = CodeSearchMatcher(search_output, Granularity.METHOD)
    _ = csm.match_against_df(df_code, directory=repo_path)
    csm.rank_matches()
    snips = csm.get_ranked_snippets()

    # Prepare code builder with matched snippets
    cbd = CodeBuilder(df_code)
    cbd.populate_snippets(snips)

    # Generate fixes using LLM
    fixer_msg = FIXER_PROMPT.generate_messages(
        user_kwargs=base_kwargs | {"snippets": snips}
    )

    try:
        txtgen.set_messages(fixer_msg)
        txtgen.prepare_input()
        txtgen.generate(max_new_tokens=9999)
        fixer_output = txtgen.get_response()
    except Exception:
        return None
    
    # Process fixes and generate unified diff
    cbd.process_snippets(fixer_output)
    paths = set([s.file_path for s in snips])

    mods = ""
    for path in paths:
        mods += cbd.get_modifications_diff(file_path=path, root_dir=repo_path)

    return mods
