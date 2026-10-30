import numpy as np

from urlformat_audit import audit, compare, factorial_variants, url_parts, variant, FormatRule, format_features
from urlformat_audit.cli import main


def test_url_parts():
    p = url_parts("http://www.Example.org/a/b/")
    assert (p.scheme, p.host, p.www, p.tail, p.trailing_slash) == ("http", "Example.org", True, "/a/b", True)
    p = url_parts("example.org")
    assert p.scheme is None and not p.has_path and not p.trailing_slash
    assert url_parts("https://a.com/").trailing_slash and not url_parts("https://a.com/").has_path


def test_factorial():
    v = factorial_variants("http://x.org/a/")
    assert len(v) == 24 and len({c for c, _ in v}) == 24
    assert variant("http://x.org/a/b/", "https", True, False, False) == "https://www.x.org"
    assert variant("x.org", "none", False, True, True) == "x.org/"


def test_audit_detects_template_shortcut():
    legit = [f"https://www.site{i}.com" for i in range(300)]
    phish = [f"http://bad{i}.xyz/login/" for i in range(300)]
    r = audit(legit + phish, [0] * 300 + [1] * 300)
    assert r.format_rule["mcc"] == 1.0 and r.template_rule["mcc"] == 1.0
    assert any("format-only rule" in f for f in r.flags)
    assert abs(r.nmi_cell_label - 1.0) < 1e-9


def test_compare_and_rule():
    tr = ["https://www.a.com", "http://b.com/x/"] * 50
    ytr = [0, 1] * 50
    te = ["http://www.c.edu/", "https://www.d.org"]
    res = compare(tr, ytr, te, [0, 0])
    assert 0 <= res["share_target_in_unseen_cells"] <= 1
    assert list(FormatRule.fit(tr, ytr).predict(["https://www.z.com"])) == [0]


def test_features_columns():
    f = format_features(["https://1.2.3.4:8080/x?q=1"])
    assert bool(f.ip_host[0]) and bool(f.port[0]) and bool(f["query"][0])


def test_cli(tmp_path, capsys):
    p = tmp_path / "d.csv"
    p.write_text("url,label\nhttps://www.a.com,good\nhttp://b.com/x/,bad\nhttps://www.c.com,good\nhttp://d.com/,bad\n")
    assert main(["audit", str(p), "--positive", "bad"]) == 0
    assert "Format-only rule" in capsys.readouterr().out
    out = tmp_path / "f.csv"
    assert main(["factorial", str(p), "-o", str(out)]) == 0
    assert out.read_text().count("\n") == 4 * 24 + 1
