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
    df: pd.DataFrame,
    root_dir: Path,
    issue_index: int,
    model_name: str,
) -> str | None:
    """
    """

    repo_id = df["instance_id"].iloc[issue_index]
    repo_name = df["repo"].iloc[issue_index]
    repo_path = root_dir / f"data/repos/repo__{repo_id}"
    problem = df["problem_statement"].iloc[issue_index]

    base_kwargs = {"repo_name": repo_name, "problem": problem}
    search_msg = SEARCH_PROMPT.generate_messages(user_kwargs=base_kwargs)
    txtgen = TextGenerator(model_name)

    try:
        txtgen.set_messages(search_msg)
        txtgen.prepare_input()
        txtgen.generate()
        search_output = txtgen.get_response()
    except:
        return
    
    df_code = analyze_codebase(directory=repo_path)
    csm = CodeSearchMatcher(search_output, Granularity.METHOD)
    _ = csm.match_against_df(df_code, directory=repo_path)
    csm.rank_matches()
    snips = csm.get_ranked_snippets()

    cbd = CodeBuilder(df_code)
    cbd.populate_snippets(snips)

    fixer_msg = FIXER_PROMPT.generate_messages(
        user_kwargs=base_kwargs | {"snippets": snips}
    )

    try:
        txtgen.set_messages(fixer_msg)
        txtgen.prepare_input()
        txtgen.generate(max_new_tokens=9999)
        fixer_output = txtgen.get_response()
    except:
        return None
    
    cbd.process_snippets(fixer_output)
    paths = set([s.file_path for s in snips])

    mods = ""
    for path in paths:
        mods += cbd.get_modifications_diff(file_path=path, root_dir=repo_path)

    return mods
