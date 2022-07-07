from typing import List, Union
import os

PROG_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.realpath(os.path.join(PROG_DIR, ".."))

class LocalPath:
    """ Represents a file or directory path in the project directory.
    Fields:
    - __project_relative_path: Path represented relative to the project directory root
    """
    __project_relative_path: str

    def __init__(self, parts: Union[List[str], str]):
        """ Initializes a LocalPath.
        Arguments:
        - parts: List of path parts, relative to the project root
        """
        parts_list = []
        if isinstance(parts, list):
            parts_list = parts
        else:
            parts_list = [ parts ]
        
        self.__project_relative_path = os.path.relpath(os.path.join(*parts_list), REPO_DIR)

    def get_project_relative_path(self) -> str:
        """ Returns: Path relative to the project root directory.
        """
        return self.__project_relative_path

    def get_absolute_path(self) -> str:
        """ Returns: Path relative to file system root.
        """
        return os.path.realpath(self.get_project_relative_path())

    def join(self, parts: List[str]) -> LocalPath:
        """ Constructs a new LocalPath with parts added onto the end.
        Arguments:
        - parts: File path parts to append.

        Returns: New LocalPath.
        """
        return LocalPath([self.get_project_relative_path(), *parts])
