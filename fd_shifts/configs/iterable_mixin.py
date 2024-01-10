from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class _IterableMixin:  # pylint: disable=too-few-public-methods
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return filter(
            lambda item: not item[0].startswith("__"), self.__dict__.items()
        ).__iter__()
