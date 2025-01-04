import os
from pathlib import Path
from typing import List, Optional

def map_directory(directory: str, indent: str = "  ", prefix: str = "") -> str:
    """
    Maps out the structure of a directory and returns it as a formatted string.
    
    Args:
        directory (str): Path to the directory to map
        indent (str): Indentation string for each level (default: two spaces)
        prefix (str): Prefix for the current line (used in recursion)
        
    Returns:
        str: A formatted string representing the directory structure
    """
    result = []
    try:
        # Convert to Path object for better path handling
        path = Path(directory)
        
        # Get all items in the directory
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        for item in items:
            # Add the current item to the result
            result.append(f"{prefix}{item.name}")
            
            # If it's a directory, recursively map its contents
            if item.is_dir():
                subdir_content = map_directory(
                    str(item),
                    indent=indent,
                    prefix=prefix + indent
                )
                result.append(subdir_content)
                
        return "\n".join(filter(None, result))
    except Exception as e:
        return f"Error mapping directory: {str(e)}"

def save_map_to_file(directory: str, output_file: str, indent: str = "  ") -> bool:
    """
    Maps a directory and saves the result to a file.
    
    Args:
        directory (str): Path to the directory to map
        output_file (str): Path where to save the map
        indent (str): Indentation string for each level
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        map_content = map_directory(directory, indent)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(map_content)
        return True
    except Exception:
        return False
