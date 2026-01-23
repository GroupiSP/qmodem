from pathlib import Path
from typing import List, Union


def mkdir_if_not_existent(paths: List[Union[str, Path]]) -> None:
    """Iterates through a list of paths and creates the directories if they do not
    already exist.

    Args:
        paths (List[Union[str, Path]]): A list of directory paths.
            Accepts both strings and Path objects.
    """
    for path in paths:
        try:
            # Convert to Path object to ensure compatibility
            dir_path = Path(path)

            # parents=True: Creates missing parent directories (e.g., creates 'a' for 'a/b/c')
            # exist_ok=True: Does not raise an error if the directory already exists
            dir_path.mkdir(parents=True, exist_ok=True)

            print(f"Checked/Created: {dir_path.resolve()}")

        except OSError as e:
            print(f"Error creating directory '{path}': {e}")
