import json
from pathlib import Path
from typing import Union, Dict


def map_directory(
    directory: str,
    prefix: str = "",
    is_last: bool = True,
    parent_prefix: str = "",
) -> str:
    """
    Maps out the structure of a directory and returns it as a formatted string
    in tree format.

    Args:
        directory (str): Path to the directory to map
        prefix (str): Prefix for the current line (used in recursion)
        is_last (bool): Whether this is the last item in its level
        parent_prefix (str): Prefix from parent level for proper line drawing

    Returns:
        str: A formatted string representing the directory structure in tree
            format
    """
    result = []
    try:
        path = Path(directory)
        items = sorted(
            path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
        )

        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1

            # Determine the current item's prefix
            if prefix == "":  # Root level
                current_prefix = ""
            else:
                current_prefix = "└── " if is_last_item else "├── "

            # Add the current item to the result
            result.append(f"{parent_prefix}{current_prefix}{item.name}")

            # If it's a directory, recursively map its contents
            if item.is_dir():
                # Calculate the new parent prefix for the next level
                new_parent_prefix = parent_prefix
                if prefix != "":  # Not root level
                    new_parent_prefix += "    " if is_last_item else "│   "

                subdir_content = map_directory(
                    str(item),
                    prefix="│   ",
                    is_last=is_last_item,
                    parent_prefix=new_parent_prefix,
                )
                if subdir_content:
                    result.append(subdir_content)

        return "\n".join(filter(None, result))
    except Exception as e:
        return f"Error mapping directory: {str(e)}"


def map_directory_json(
    directory: str,
    as_string: bool = False
) -> Union[Dict, str]:
    """
    Maps out the structure of a directory and returns it as a JSON object
    or formatted JSON string.

    Args:
        directory (str): Path to the directory to map
        as_string (bool): If True, returns a formatted JSON string instead
            of a Python dictionary

    Returns:
        Union[Dict, str]: A dictionary or JSON string representing the
            directory structure
    """
    def _build_tree(path: Path) -> Dict:
        result = {}
        try:
            items = sorted(
                path.iterdir(),
                key=lambda x: (not x.is_dir(), x.name.lower())
            )
            
            for item in items:
                if item.is_dir():
                    result[item.name] = _build_tree(item)
                else:
                    result[item.name] = None
                    
            return result
        except Exception as e:
            return {"error": str(e)}
    
    try:
        tree = _build_tree(Path(directory))
        if as_string:
            return json.dumps(tree, indent=2)
        return tree
    except Exception as e:
        error_result = {"error": f"Error mapping directory: {str(e)}"}
        if as_string:
            return json.dumps(error_result)
        return error_result
