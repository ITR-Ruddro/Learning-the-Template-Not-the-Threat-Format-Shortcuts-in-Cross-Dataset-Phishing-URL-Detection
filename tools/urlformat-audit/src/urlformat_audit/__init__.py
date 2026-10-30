"""urlformat-audit: detect class-conditional URL formatting (shortcut risk) in labelled URL datasets."""
__version__ = "0.1.0"

from .parse import url_parts, format_features, cell_of  # noqa: F401
from .audit import audit, compare, binary_metrics, FormatRule, TEMPLATE_REGEX  # noqa: F401
from .factorial import factorial_variants, variant, expand, CELLS  # noqa: F401
