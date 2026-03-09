"""llm-assert has been renamed to callspec.

This shim package prints a deprecation warning and re-exports callspec
so existing code continues working without changes. Update your imports
to use callspec directly.
"""

import warnings

warnings.warn(
    "llm-assert has been renamed to callspec. "
    "Install with: pip install callspec. "
    "Update imports: from callspec import Callspec",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export callspec so existing code keeps working
from callspec import *  # noqa: F401, F403, E402
from callspec import Callspec as LLMAssert  # noqa: F401, E402
