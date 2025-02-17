from typing import Any, Dict, Union
import time

from .code_analyzer import analyze_codebase
from .code_builder import CodeBuilder
from .code_search import Granularity, CodeSearchMatcher
from .common import SearchMode
from .prompt import FIXER_PROMPT, SEARCH_PROMPT, SearchPromptType
from .model import MAX_TOKENS, TextGenerator


def _init_or_reuse_model(
    model_spec: Union[str, TextGenerator],
    existing_model: TextGenerator | None = None,
) -> TextGenerator:
    """Initialize a new model or reuse existing if compatible.

    Args:
        model_spec: Model name or TextGenerator instance
        existing_model: Optional existing TextGenerator to potentially reuse

    Returns:
        Initialized TextGenerator instance
    """
    if isinstance(model_spec, TextGenerator):
        if not hasattr(model_spec, "model") or model_spec.model is None:
            raise ValueError("TextGenerator is not properly initialized")
        return model_spec
    elif existing_model is not None and isinstance(model_spec, str):
        # Reuse existing model if it matches the requested model name
        if existing_model.model == model_spec:
            return existing_model
    # Create new model instance
    return TextGenerator(model_spec)


def run_pipeline(
    repo_path: str,
    problem: str,
    search_model: Union[str, TextGenerator],
    fix_model: Union[str, TextGenerator] | None = None,
    search_kwargs: Dict[str, Any] | None = None,
    search_gen_kwargs: Dict[str, Any] | None = None,
    fix_gen_kwargs: Dict[str, Any] | None = None,
    tokens_per_second: int = 5,
    verbose: bool = False,
    print_list: list[str] | None = None,
    timeout_minutes: float = 30.0,
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
        search_model: Either a model name string or a TextGenerator object
            for the search task.
        fix_model: Either a model name string or a TextGenerator object for
            the fix task. If None, search_model is reused.
        search_kwargs: Controls how the search prompt is compiled
        search_gen_kwargs: Optional kwargs for search model generation
        fix_gen_kwargs: Optional kwargs for fix model generation
        tokens_per_second: Tokens that model is expected to process per second
        verbose: If True, log the LLM responses for debugging
        print_list: Overrides verbose=False for specified items
        timeout_minutes: Maximum number of minutes for pipeline to run

    Returns:
        String containing unified diff of proposed changes, or None if it fails
    """

    # print list for verbose overrides
    print_list = print_list or []

    # Start timing
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    # Prepare common kwargs for prompts
    base_kwargs = {"problem": problem}

    # Default search kwargs if None
    search_kwargs = search_kwargs or {"prompt_type": SearchPromptType.DT}
    search_mode = search_kwargs.get("search_mode", SearchMode.LINE_AND_PARENT)

    # Initialize search model
    search_msg = SEARCH_PROMPT.generate_messages(
        user_kwargs=base_kwargs | search_kwargs,
        verbose=verbose,
    )

    search_txtgen = _init_or_reuse_model(search_model)
    search_txtgen.reset_max_tokens(MAX_TOKENS)
    search_txtgen.set_messages(search_msg)
    search_txtgen.prepare_input()

    search_txtgen.generate(**(search_gen_kwargs or {}))
    search_output = search_txtgen.get_response()
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

    # init codebuilder
    cbd = CodeBuilder(df_code)

    # Initialize or reuse model for fixing
    fix_model = fix_model or search_model  # Use search model if no fix model
    fix_txtgen = _init_or_reuse_model(
        fix_model, search_txtgen if fix_model == search_model else None
    )
    fix_txtgen.reset_max_tokens(MAX_TOKENS)

    # Check time after search
    search_time = time.time() - start_time
    remaining_time = timeout_seconds - search_time
    if verbose or ("time_check" in print_list):
        print(f"Time remaining: {remaining_time:.1f}s")
        print(
            f"Processed {search_txtgen.input_length} tokens "
            + f"in {search_time} seconds"
        )

    # review max tokens based on elapsed time
    max_tokens = int(remaining_time * tokens_per_second)
    fix_txtgen.reset_max_tokens(min(max_tokens, MAX_TOKENS))

    def _fix_prompt_gen(min_score: float) -> tuple[list, bool] | None:
        snips = csm.get_ranked_snippets(min_score)

        # Stop pipeline if not relevant snippets found
        if len(snips) == 0:
            print(f"!!! No code snippets found using min_score={min_score}")
            return None

        # Prepare code builder with matched snippets
        cbd.populate_snippets(snips)

        # Generate fixes using LLM
        fixer_msg = FIXER_PROMPT.generate_messages(
            user_kwargs=base_kwargs | {"snippets": snips}, verbose=verbose
        )

        fix_txtgen.set_messages(fixer_msg)
        fix_txtgen.prepare_input()

        return snips, fix_txtgen.prompt_tokens_over_limit

    valid_snippets = None
    for min_score in [0, 1, 2, 3]:
        result = _fix_prompt_gen(min_score)
        if result is None:
            continue

        snips, tokens_over_limit = result
        if not tokens_over_limit:
            valid_snippets = snips
            break

    if valid_snippets is None:
        print("!!! Could not find snippets that fit within token limit")
        return None

    snips = valid_snippets

    # if fix_txtgen.prompt_tokens_over_limit:
    # TODO: revisit matching df and see if we can identify children
    # if it's a larger parent causing the large token size
    # e.g. a monolithic class object

    fix_txtgen.generate(**(fix_gen_kwargs or {}))
    fixer_output = fix_txtgen.get_response()
    if verbose or ("fix_output" in print_list):
        print(">>> FIXER TASK OUTPUT START <<<\n")
        print(fixer_output)
        print("\n>>> FIXER TASK OUTPUT END <<<")

    # Check if the fix was successful
    status = fix_txtgen.parse_status(fixer_output)
    if not status == fix_txtgen.ResponseStatus.SUCCESS:
        print(f"!!! Skipping issue due to {status.name} status...")
        return None

    # Process fixes and generate unified diff
    cbd.process_snippets(fixer_output)
    paths = set([s.file_path for s in snips])

    mods = ""
    for path in paths:
        mods += cbd.get_modifications_diff(file_path=path, root_dir=repo_path)

    # Check time after fix
    fix_time = time.time() - start_time
    remaining_time = timeout_seconds - fix_time
    if verbose or ("time_check" in print_list):
        print(f"Time remaining: {remaining_time:.1f}s")
        print(
            f"Processed {fix_txtgen.input_length} tokens "
            + f"in {fix_time - search_time} seconds"
        )

    return mods
