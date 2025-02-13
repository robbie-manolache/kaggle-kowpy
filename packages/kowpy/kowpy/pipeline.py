from typing import Union

from .code_analyzer import analyze_codebase
from .code_builder import CodeBuilder
from .code_search import Granularity, CodeSearchMatcher
from .common import SearchMode
from .prompt import FIXER_PROMPT, SEARCH_PROMPT
from .model import TextGenerator


def run_pipeline(
    repo_path: str,
    problem: str,
    model: Union[str, TextGenerator],
    search_mode: SearchMode = SearchMode.LINE_ONLY,
    verbose: bool = False,
    print_list: list[str] | None = None,
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
        search_mode: Controls which JSON template to use for searching
        verbose: If True, log the LLM responses for debugging
        print_list: Overrides verbose=False for specified items

    Returns:
        String containing unified diff of proposed changes, or None if it fails
    """

    # print list for verbose overrides
    print_list = print_list or []

    # Prepare common kwargs for prompts
    base_kwargs = {"problem": problem}

    # Initialize text generator with model or validate existing one
    search_msg = SEARCH_PROMPT.generate_messages(
        user_kwargs=base_kwargs | {"search_mode": search_mode},
        verbose=verbose,
    )

    if isinstance(model, str):
        txtgen = TextGenerator(model)
    else:
        txtgen = model
        # Validate the TextGenerator instance
        if not hasattr(txtgen, "model") or txtgen.model is None:
            raise ValueError("TextGenerator is not properly initialized")

    txtgen.set_messages(search_msg)
    txtgen.prepare_input()
    txtgen.generate(max_new_tokens=512)
    search_output = txtgen.get_response()
    if verbose or ("search_output" in print_list):
        print(">>> SEARCH TASK OUTPUT START <<<\n")
        print(search_output)
        print("\n>>> SEARCH TASK OUTPUT END <<<")

    # Analyze codebase and find relevant code sections
    df_code = analyze_codebase(directory=repo_path)
    csm = CodeSearchMatcher(search_output, search_mode, Granularity.METHOD)
    _ = csm.match_against_df(df_code, directory=repo_path)
    _ = csm.rank_matches()
    if verbose or ("ranked_matches" in print_list):
        print(csm.ranked_matches_df)
    snips = csm.get_ranked_snippets()

    # Prepare code builder with matched snippets
    cbd = CodeBuilder(df_code)
    cbd.populate_snippets(snips)

    # Generate fixes using LLM
    fixer_msg = FIXER_PROMPT.generate_messages(
        user_kwargs=base_kwargs | {"snippets": snips}, verbose=verbose
    )

    txtgen.set_messages(fixer_msg)
    txtgen.prepare_input()
    if txtgen.prompt_tokens_over_limit:
        # TODO: revisit matching df and see if we can identify children
        # if it's a larger parent causing the large token size
        # e.g. a monolithic class object
        print("!!! Skipping issue due to large prompt size...")
        return None

    txtgen.generate()
    fixer_output = txtgen.get_response()
    if verbose:
        print(">>> FIXER TASK OUTPUT START <<<\n")
        print(fixer_output)
        print("\n>>> FIXER TASK OUTPUT END <<<")

    # Check if the fix was successful
    status = txtgen.parse_status(fixer_output)
    if not status == txtgen.ResponseStatus.SUCCESS:
        print(f"!!! Skipping issue due to {status.name} status...")
        return None

    # Process fixes and generate unified diff
    cbd.process_snippets(fixer_output)
    paths = set([s.file_path for s in snips])

    mods = ""
    for path in paths:
        mods += cbd.get_modifications_diff(file_path=path, root_dir=repo_path)

    return mods
