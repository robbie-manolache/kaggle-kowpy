import logging
from typing import Union

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
    model: Union[str, TextGenerator],
    verbose: bool = False,
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
        repo_path: Path to the repository to analyze
        problem: Description of the problem to fix
        model: Either a model name string or an initialized TextGenerator
        verbose: If True, log the LLM responses for debugging

    Returns:
        String containing unified diff of proposed changes, or None if it fails
    """

    # Prepare common kwargs for prompts
    base_kwargs = {"problem": problem}

    # Initialize text generator with model or validate existing one
    search_msg = SEARCH_PROMPT.generate_messages(user_kwargs=base_kwargs)

    if isinstance(model, str):
        txtgen = TextGenerator(model)
    else:
        txtgen = model
        # Validate the TextGenerator instance
        if not hasattr(txtgen, "model") or txtgen.model is None:
            logging.error("TextGenerator instance is not properly initialized")
            return None

    try:
        txtgen.set_messages(search_msg)
        txtgen.prepare_input()
        txtgen.generate(max_new_tokens=1024)
        search_output = txtgen.get_response()
        if verbose:
            logging.info(f"Search criteria generated:\n{search_output}")
    except Exception as e:
        logging.error(f"Error generating search criteria: {str(e)}")
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
        if verbose:
            logging.info(f"Fix suggestions generated:\n{fixer_output}")
    except Exception as e:
        logging.error(f"Error generating fixes: {str(e)}")
        return None

    # Process fixes and generate unified diff
    cbd.process_snippets(fixer_output)
    paths = set([s.file_path for s in snips])

    mods = ""
    for path in paths:
        mods += cbd.get_modifications_diff(file_path=path, root_dir=repo_path)

    return mods
