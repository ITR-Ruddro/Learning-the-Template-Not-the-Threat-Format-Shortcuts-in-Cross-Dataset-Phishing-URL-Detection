"""Factorial format edits: rebuild a URL in every combination of scheme x www x trailing slash x path."""
from __future__ import annotations

from itertools import product
from typing import Iterable, List, Tuple

import pandas as pd

from .parse import url_parts

FACTOR_SCHEMES = ("https", "http", "none")
CELLS: List[Tuple[str, bool, bool, bool]] = [(s, w, sl, p) for s, w, sl, p in product(FACTOR_SCHEMES, (True, False), (True, False), (True, False))]


def cell_name(scheme: str, www: bool, slash: bool, path: bool) -> str:
    return "|".join([scheme, "www" if www else "nowww", "slash" if slash else "noslash", "path" if path else "nopath"])


def variant(url: str, scheme: str, www: bool, slash: bool, path: bool) -> str:
    """Rebuild ``url`` with the given format; the path is everything after the host.

    >>> variant("http://x.org/a/b/", "https", True, False, False)
    'https://www.x.org'
    """
    p = url_parts(url)
    head = "" if scheme == "none" else f"{scheme}://"
    body = ("www." if www else "") + p.host + (p.tail if path else "")
    return head + body + ("/" if slash else "")


def factorial_variants(url: str) -> List[Tuple[str, str]]:
    """All 24 (cell name, edited URL) pairs for one URL."""
    return [(cell_name(*c), variant(url, *c)) for c in CELLS]


def expand(urls: Iterable[str], ids: Iterable = None) -> pd.DataFrame:
    """Long table (id, cell, url_variant) for scoring with any model."""
    urls = list(urls)
    ids = list(ids) if ids is not None else list(range(len(urls)))
    rows = [(i, c, v) for i, u in zip(ids, urls) for c, v in factorial_variants(u)]
    return pd.DataFrame(rows, columns=["id", "cell", "url_variant"])
