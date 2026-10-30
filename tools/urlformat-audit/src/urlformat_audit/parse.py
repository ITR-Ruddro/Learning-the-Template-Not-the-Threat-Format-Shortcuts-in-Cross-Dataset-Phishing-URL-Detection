"""Parsing of the format properties of URL strings (no network access, no external parsers)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

SCHEMES = ("https", "http", "none")  # 'other' schemes (ftp, ...) are mapped to 'other' in features


@dataclass(frozen=True)
class URLParts:
    scheme: Optional[str]   # lower-cased scheme or None
    host: str               # host[:port] without a leading 'www.' (case kept)
    www: bool               # host started with 'www.'
    tail: str               # everything after the host (path, query, fragment) without a trailing '/'
    trailing_slash: bool    # the string ended with '/' after something non-empty

    @property
    def has_path(self) -> bool:
        return len(self.tail) > 0


def _split_host(rest: str) -> tuple[str, str]:
    cuts = [i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i >= 0]
    cut = min(cuts) if cuts else len(rest)
    return rest[:cut], rest[cut:]


def url_parts(url: str) -> URLParts:
    """Split a URL string into the parts used by the audit.

    >>> url_parts("http://www.Example.org/a/b/")
    URLParts(scheme='http', host='Example.org', www=True, tail='/a/b', trailing_slash=True)
    """
    u = str(url).strip()
    m = _SCHEME.match(u)
    scheme = m.group(1).lower() if m else None
    rest = u[m.end():] if m else u
    host, tail = _split_host(rest)
    www = host.lower().startswith("www.")
    if www:
        host = host[4:]
    slash = (host + tail).endswith("/") and len((host + tail).rstrip("/")) > 0
    if slash:
        tail = tail.rstrip("/")
    return URLParts(scheme, host, www, tail, slash)


def scheme_group(scheme: Optional[str]) -> str:
    if scheme is None:
        return "none"
    return scheme if scheme in ("http", "https") else "other"


def cell_of(url: str) -> str:
    """Format cell 'scheme|www|slash|path' used by the audit and the factorial edits."""
    p = url_parts(url)
    return "|".join([scheme_group(p.scheme), "www" if p.www else "nowww",
                     "slash" if p.trailing_slash else "noslash", "path" if p.has_path else "nopath"])


def format_features(urls: Iterable[str]) -> pd.DataFrame:
    """One row per URL with boolean/numeric format features and its format cell."""
    rows = []
    for u in urls:
        s = str(u).strip()
        p = url_parts(s)
        hostname = p.host.split(":")[0]
        rows.append({
            "scheme": scheme_group(p.scheme),
            "www": p.www,
            "trailing_slash": p.trailing_slash,
            "path": p.has_path,
            "query": "?" in p.tail,
            "port": bool(re.search(r":\d+$", p.host)),
            "ip_host": bool(_IPV4.match(hostname)) or hostname.startswith("["),
            "uppercase": any(c.isupper() for c in s),
            "length": len(s),
            "cell": "|".join([scheme_group(p.scheme), "www" if p.www else "nowww",
                              "slash" if p.trailing_slash else "noslash", "path" if p.has_path else "nopath"]),
        })
    return pd.DataFrame(rows)
