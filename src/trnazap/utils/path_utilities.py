from pathlib import Path
from typing import Iterable, List, Union, Optional, Collection, Set
import glob

class PathSet:
    def __init__(self, paths: Optional[Iterable[Union[str, Path]]]=None):
        if paths:
            self._paths = {Path(p).resolve() for p in paths}
        else:
            self._paths = set()

    def __contains__(self, path: Union[str, Path]) -> bool:
        return Path(path).resolve() in self._paths

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def __or__(self, other: "PathSet") -> "PathSet":
        return PathSet(self._paths | other._paths)

    def __and__(self, other: "PathSet") -> "PathSet":
        return PathSet(self._paths & other._paths)

    def __sub__(self, other: "PathSet") -> "PathSet":
        return PathSet(self._paths - other._paths)
    
    def __add__(self, other: "PathSet") -> "PathSet":
        return PathSet(self._paths.union(other._paths))

    def add(self, path: Union[str, Path]) -> None:
        self._paths.add(Path(path).resolve())

    def to_list(self) -> List[str]:
        return [str(p) for p in sorted(self._paths)]

    @classmethod
    def from_list(cls, paths: List[str]) -> "PathSet":
        return cls(paths)
    
    @property
    def paths(self):
        return self._paths

    def __repr__(self):
        return f"PathSet({self.to_list()})"
    


def search_path(path: Path, recursive: bool, patterns: Collection[str]) -> Set[Path]:
    """
    Search `path` matching `pattern` searching directories recursively if requested
    """

    def _any_match(path: Path):
        return any(path.match(p) for p in patterns)

    # Get the recursive or non-recursive glob function.
    matching_files = set()
    if path.is_dir():
        pattern = str(path / "**" / "*") if recursive else str(path / "*")
        for matching_pathname in glob.glob(pattern, recursive=recursive):
            matching_path = Path(matching_pathname)
            if matching_path.is_file() and _any_match(matching_path):
                matching_files.add(matching_path)

    # Non-directory, assert that it is a file and that it matches the file_pattern
    elif path.is_file() and _any_match(path):
        matching_files.add(path)

    return matching_files

