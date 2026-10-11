from raglab.api.security import write_allowed

ALLOWED = ["http://localhost:5173"]


def test_non_browser_clients_send_no_origin_and_are_allowed() -> None:
    assert write_allowed(None, host="127.0.0.1:8000", allowed=ALLOWED)


def test_listed_and_same_origin_writes_are_allowed() -> None:
    assert write_allowed("http://localhost:5173", host="127.0.0.1:8000", allowed=ALLOWED)
    assert write_allowed("http://127.0.0.1:8000", host="127.0.0.1:8000", allowed=ALLOWED)


def test_other_origins_are_refused() -> None:
    assert not write_allowed("https://evil.example", host="127.0.0.1:8000", allowed=ALLOWED)
    assert not write_allowed("null", host="127.0.0.1:8000", allowed=ALLOWED)
    assert not write_allowed("http://localhost:8000", host="127.0.0.1:8000", allowed=ALLOWED)
