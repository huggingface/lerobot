"""Compatibility import for the Muon implementation now located in ``muon``.

Alias the module, rather than copying its exports, so private helpers, old
pickle references and monkeypatches resolve to the same implementation.
"""

import sys

from . import muon

sys.modules[__name__] = muon
