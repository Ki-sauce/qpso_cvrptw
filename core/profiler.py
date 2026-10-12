import time
from functools import wraps
from contextlib import contextmanager
import logging

logger = logging.getLogger("uvicorn.error")

class RequestProfiler:
    def __init__(self):
        self.timings = {}
        self.start_time = None

    def start_request(self):
        self.timings.clear()
        self.start_time = time.perf_counter()

    @contextmanager
    def measure(self, block_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.timings[block_name] = elapsed

    def summary(self):
        if not self.start_time:
            return "Profiler not started."
        total_time = time.perf_counter() - self.start_time
        
        lines = [
            "\n==============================================",
            "          REQUEST PIPELINE PROFILER           ",
            "=============================================="
        ]
        for name, duration in self.timings.items():
            pct = (duration / total_time) * 100 if total_time > 0 else 0
            lines.append(f"  • {name:<35} : {duration:>7.4f}s ({pct:>5.1f}%)")
        lines.append("-" * 46)
        lines.append(f"  • {'TOTAL REQUEST TIME':<35} : {total_time:>7.4f}s")
        lines.append("==============================================\n")
        
        return "\n".join(lines)

# Global or request-local profiler instance
profiler = RequestProfiler()