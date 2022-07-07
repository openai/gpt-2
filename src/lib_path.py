from typing import List, Union
import os
from pathlib import PureWindowsPath, PurePosixPath

PROG_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.realpath(os.path.join(PROG_DIR, ".."))

class UnableToReconcileAbsPathError(Exception):
    """ Indicates that an absolute path was received but it couldn't be turned into a relative path.
    Specifically the following conditions must be met:
    - an absolute path was received
    - but that absolute path was determined not to be from this machine's file system
    - and an effort was undertaken to strip the parts of the abolute path which were not from this machine's file system
    - but not a single part of the absolute path was found to be from this project's directory structure

    Fields:
    - raw_path: The unprocessed path
    - os_path: The path as represented by the pathlib primitive for the paths origin operating system
    """
    raw_path: str
    os_path: Union[PurePosixPath, PureWindowsPath]

    def __init__(self, raw_path: str, os_path: Union[PurePosixPath, PureWindowsPath]):
        """ Initialize a UnabletoReconcileAbsPathError.
        Arguments:
        - raw_path: See UnableToReconcileAbsPathError.raw_path field
        - os_path: See UnableToReconcileAbsPathError.os_path field
        """
        super().__init__(f"The absolute path '{raw_path}' was determined to not be from this device's file system and not from this project's directory structure")
        self.raw_path = raw_path
        self.os_path = os_path

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

        If parts happens to be an absolute path from another machine a best attempt is made to reconcile the path into something relative to the project's directory. This behavior exists because previous versions of the code saved absolute paths in metadata files, making them non-portable.

        Raises:
        - UnableToReconcileAbsPathError: If the absolute path cannot be reconciled, see error class documentation for more details
        """
        # Join parts together
        joined_parts = ""
        if isinstance(parts, list):
            joined_parts = os.path.join(*parts)
        else:
            joined_parts = parts

        # Fix absolute paths from other systems, this is a regression introduced in previous versions of the code where absolute paths were stored in metadata files
        # First determine if the path is formatted in a Windows format or a Posix format
        posix_path_parse = PurePosixPath(joined_parts)
        windows_path_parse = PureWindowsPath(joined_parts)

        # Absolute Posix paths will not be determined as absolute paths on Windows and vice versa
        # Therefore we need to determine what operating system this path was created on and check if it is absolute appropriately
        is_absolute = False
        path_from_os = None

        if posix_path_parse.as_posix() != windows_path_parse.as_posix():
            # A Windows absolute path fed into both the posix and windows path class and then converted back to a posix path will result in different return values from each respective class
            # However a posix path put into both classes will result in the same return value
            # Therefore this is a windows path
            is_absolute = windows_path_parse.is_absolute()
            path_from_os = windows_path_parse
        else:
            # This is a posix path because it did not fit the criteria described in the doc comments in the other branch
            is_absolute = posix_path_parse.is_absolute()
            path_from_os = posix_path_parse

        if is_absolute:
            # Determine if absolute path is from another machine
            if not path_from_os.is_relative_to(PROG_DIR):
                # Path not from this machine, try to remove part of path that references other machine
                # First try to build by appending all non file (nf) name parts
                parts = list(path_from_os.parts)
                acceptable_parts_nf = [ parts.pop() ]

                while os.path.splitext(parts[len(parts) - 1])[1] != "":
                    acceptable_parts_nf.append(parts.pop())

                # Then try to build by finding existing (e) files and directories
                parts = list(path_from_os.parts)[1:]
                acceptable_parts_e = []

                # Add parts to the list if they point towards existing files
                while len(parts) > 0 or len(acceptable_parts_e) == 0:
                    part_zero = parts.pop(0)
                    if os.path.exists(os.path.join(REPO_DIR, *[*acceptable_parts_e, part_zero])):
                        acceptable_parts_e.append(part_zero)

                # Use the solution which found the longest path
                if len(acceptable_parts_nf) + len(acceptable_parts_e) > 0:
                    if len(acceptable_parts_nf) > len(acceptable_parts_e):
                        joined_parts = os.path.join(*acceptable_parts_nf)
                    else:
                        joined_parts = os.path.join(*acceptable_parts_e)

                
        # Save relative path
        self.__project_relative_path = os.path.relpath(joined_parts, REPO_DIR)

    def get_project_relative_path(self) -> str:
        """ Returns: Path relative to the project root directory.
        """
        return self.__project_relative_path

    def get_absolute_path(self) -> str:
        """ Returns: Path relative to file system root.
        """
        return os.path.realpath(self.get_project_relative_path())

    def join(self, parts: List[str]) -> 'LocalPath':
        """ Constructs a new LocalPath with parts added onto the end.
        Arguments:
        - parts: File path parts to append.

        Returns: New LocalPath.
        """
        return LocalPath([self.get_project_relative_path(), *parts])
