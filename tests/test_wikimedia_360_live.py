from __future__ import annotations

import unittest
from email.message import Message
from urllib.error import HTTPError

from processing.wikimedia_360_live import _with_backoff


def _headers(retry_after: str | None = None) -> Message:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


class Wikimedia360LiveTest(unittest.TestCase):
    def test_retries_429_using_retry_after(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise HTTPError(
                    "https://commons.wikimedia.org/w/api.php",
                    429,
                    "Too Many Requests",
                    _headers("0.25"),
                    None,
                )
            return {"ok": True}

        result = _with_backoff(operation, sleep=sleeps.append)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)
        self.assertEqual(sleeps, [0.25])

    def test_retries_503_with_exponential_fallback(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise HTTPError(
                    "https://commons.wikimedia.org/w/api.php",
                    503,
                    "Service Unavailable",
                    _headers(),
                    None,
                )
            return {"ok": True}

        result = _with_backoff(operation, sleep=sleeps.append)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [5.0, 10.0])

    def test_retries_maxlag_runtime_error(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Wikimedia API error: {'code': 'maxlag'}")
            return {"ok": True}

        result = _with_backoff(operation, sleep=sleeps.append)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)
        self.assertEqual(sleeps, [5.0])

    def test_non_rate_limit_http_error_fails_without_retry(self) -> None:
        sleeps: list[float] = []

        def operation():
            raise HTTPError(
                "https://commons.wikimedia.org/w/api.php",
                404,
                "Not Found",
                _headers(),
                None,
            )

        with self.assertRaises(HTTPError):
            _with_backoff(operation, sleep=sleeps.append)

        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()
