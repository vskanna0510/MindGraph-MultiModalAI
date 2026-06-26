#!/usr/bin/env python3
"""Basic API latency benchmark for MindGraph++ health endpoint."""

import statistics
import time
import urllib.error
import urllib.request


def main() -> None:
    url = "http://localhost:8000/api/v1/health"
    latencies: list[float] = []
    iterations = 20

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                response.read()
        except urllib.error.URLError as exc:
            print(f"Request failed: {exc}")
            return
        latencies.append((time.perf_counter() - start) * 1000)

    print(f"Iterations: {iterations}")
    print(f"p50: {statistics.median(latencies):.2f} ms")
    print(f"mean: {statistics.mean(latencies):.2f} ms")
    print(f"max: {max(latencies):.2f} ms")


if __name__ == "__main__":
    main()
