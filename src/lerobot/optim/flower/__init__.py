# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

SCHEDULER_REGISTRY = {}
SCHEDULER_DATACLASS_REGISTRY = {}


def register_scheduler(name: str, dataclass=None):
    """
    New scheduler types can be added to OpenSpeech with the :func:`register_scheduler` function decorator.

    For example::
        @register_scheduler('reduce_lr_on_plateau')
        class ReduceLROnPlateau:
            (...)

    .. note:: All scheduler must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the scheduler
    """

    def register_scheduler_cls(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Cannot register duplicate scheduler ({name})")

        SCHEDULER_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in SCHEDULER_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate scheduler ({name})")
            SCHEDULER_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_scheduler_cls



@dataclass
class OpenspeechDataclass:
    """OpenSpeech base dataclass that supported fetching attributes and metas"""

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(self, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")
    


@dataclass
class LearningRateSchedulerConfigs(OpenspeechDataclass):
    """Super class of learning rate dataclass"""

    lr: float = field(default=1e-04, metadata={"help": "Learning rate"})
    