from pathlib import Path
from typing import Iterable, List, Union, Optional

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

