import math
import re
import threading
import time
from collections import deque
from typing import Optional, Union

from fastapi import Request
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from .model_selection import ModelConfig

METRICS_SAMPLE_SIZE = 1000
METRICS_RATE_WINDOW_SECONDS = 60


class ProcessInfo(BaseModel):
    model: str
    token_size: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processed_time: float
    request_validation_time: float
    model_setup_time: float
    prompt_preparation_time: float
    llm_processing_time: float
    response_processing_time: float
    api_processing_time: float
    total_processing_time: float
    tokens_per_second: Optional[float]
    estimated_cost_usd: Optional[float]
    # How the agent chose the model (e.g. "Chosen by AI agent: ... <reason>").
    model_selection: Optional[str] = None


class RuntimeMetrics:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.recent_requests_by_second: deque[tuple[int, int]] = deque()
        self.completed_requests: deque[tuple[float, float]] = deque(
            maxlen=METRICS_SAMPLE_SIZE
        )
        self.process_metrics: dict[str, dict[str, Union[int, float]]] = {}

    def request_started(self) -> None:
        current_second = int(time.perf_counter())
        with self.lock:
            self.request_count += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if (
                self.recent_requests_by_second
                and self.recent_requests_by_second[-1][0] == current_second
            ):
                second, count = self.recent_requests_by_second[-1]
                self.recent_requests_by_second[-1] = (second, count + 1)
            else:
                self.recent_requests_by_second.append((current_second, 1))
            while (
                self.recent_requests_by_second
                and self.recent_requests_by_second[0][0] < current_second - 59
            ):
                self.recent_requests_by_second.popleft()

    def request_finished(self, status_code: int, elapsed: float) -> None:
        with self.lock:
            self.in_flight -= 1
            if status_code >= 500:
                self.error_count += 1
            self.completed_requests.append((time.perf_counter(), elapsed))

    def record_process(self, process: str, succeeded: bool, elapsed: float) -> None:
        with self.lock:
            metrics = self.process_metrics.setdefault(
                process,
                {
                    "run_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_elapsed_seconds": 0.0,
                    "last_elapsed_seconds": 0.0,
                    "max_elapsed_seconds": 0.0,
                },
            )
            metrics["run_count"] += 1
            metrics["success_count" if succeeded else "failure_count"] += 1
            metrics["total_elapsed_seconds"] += elapsed
            metrics["last_elapsed_seconds"] = elapsed
            metrics["max_elapsed_seconds"] = max(
                metrics["max_elapsed_seconds"], elapsed
            )

    def snapshot(self) -> dict[str, Union[int, float, str]]:
        now = time.perf_counter()
        with self.lock:
            samples = list(self.completed_requests)
            uptime = now - self.started_at
            request_count = self.request_count
            error_count = self.error_count
            in_flight = self.in_flight
            max_in_flight = self.max_in_flight
            recent_request_count = sum(
                count
                for second, count in self.recent_requests_by_second
                if second >= int(now) - 59
            )
            process_metrics = {
                process: metrics.copy()
                for process, metrics in self.process_metrics.items()
            }

        latencies = sorted(elapsed for _, elapsed in samples)
        sample_count = len(latencies)
        sample_average = sum(latencies) / sample_count if sample_count else 0.0
        sample_p50 = (
            (latencies[(sample_count - 1) // 2] + latencies[sample_count // 2]) / 2
            if sample_count
            else 0.0
        )
        sample_p95 = (
            latencies[math.ceil(sample_count * 0.95) - 1]
            if sample_count
            else 0.0
        )

        snapshot: dict[str, Union[int, float, str]] = {
            "scope": "Single API worker process; resets when the process restarts.",
            "uptime_seconds": round(uptime, 3),
            "request_count_since_start": request_count,
            "server_error_count_since_start": error_count,
            "server_error_rate_percent_since_start": (
                round(error_count / request_count * 100, 2)
                if request_count
                else 0.0
            ),
            "in_flight_requests": in_flight,
            "max_concurrent_requests_observed": max_in_flight,
            "completed_latency_sample_size": sample_count,
            "latency_sample_limit": METRICS_SAMPLE_SIZE,
            "average_latency_seconds_last_sample": round(sample_average, 3),
            "p50_latency_seconds_last_sample": round(sample_p50, 3),
            "p95_latency_seconds_last_sample": round(sample_p95, 3),
            "requests_last_60_seconds": recent_request_count,
            "observed_requests_per_second_last_60_seconds": round(
                recent_request_count / METRICS_RATE_WINDOW_SECONDS,
                3,
            ),
            "scalability_note": (
                "Observed single-process runtime metrics, not a capacity or "
                "multi-worker scalability benchmark."
            ),
        }
        for process, metrics in process_metrics.items():
            process_name = re.sub(r"[^a-z0-9]+", "_", process.lower()).strip("_")
            prefix = f"process_{process_name}"
            run_count = int(metrics["run_count"])
            snapshot[f"{prefix}_run_count"] = run_count
            snapshot[f"{prefix}_success_count"] = int(metrics["success_count"])
            snapshot[f"{prefix}_failure_count"] = int(metrics["failure_count"])
            snapshot[f"{prefix}_average_elapsed_seconds"] = round(
                float(metrics["total_elapsed_seconds"]) / run_count, 3
            )
            snapshot[f"{prefix}_last_elapsed_seconds"] = round(
                float(metrics["last_elapsed_seconds"]), 3
            )
            snapshot[f"{prefix}_max_elapsed_seconds"] = round(
                float(metrics["max_elapsed_seconds"]), 3
            )
        return snapshot


runtime_metrics = RuntimeMetrics()


async def collect_runtime_metrics(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    if request.url.path == "/api/metrics":
        return await call_next(request)

    started_at = time.perf_counter()
    runtime_metrics.request_started()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        runtime_metrics.request_finished(
            status_code,
            time.perf_counter() - started_at,
        )


def _usage_int(usage: object, key: str) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _process_info(
    model_config: ModelConfig,
    raw_response: object,
    request_validation_time: float,
    model_setup_time: float,
    prompt_preparation_time: float,
    llm_processing_time: float,
    response_processing_time: float,
) -> ProcessInfo:
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    response_metadata = getattr(raw_response, "response_metadata", None)
    token_usage = (
        response_metadata.get("token_usage")
        if isinstance(response_metadata, dict)
        else None
    )

    input_tokens = _usage_int(usage_metadata, "input_tokens")
    if input_tokens is None:
        input_tokens = _usage_int(token_usage, "prompt_tokens")

    output_tokens = _usage_int(usage_metadata, "output_tokens")
    if output_tokens is None:
        output_tokens = _usage_int(token_usage, "completion_tokens")

    token_size = _usage_int(usage_metadata, "total_tokens")
    if token_size is None:
        token_size = _usage_int(token_usage, "total_tokens")
    if token_size is None and input_tokens is not None and output_tokens is not None:
        token_size = input_tokens + output_tokens

    cost = token_usage.get("cost") if isinstance(token_usage, dict) else None
    estimated_cost = (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0
        else None
    )
    tokens_per_second = (
        output_tokens / llm_processing_time
        if output_tokens is not None and llm_processing_time > 0
        else None
    )

    return ProcessInfo(
        model=model_config.name,
        token_size=token_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        processed_time=round(llm_processing_time, 3),
        request_validation_time=round(request_validation_time, 3),
        model_setup_time=round(model_setup_time, 3),
        prompt_preparation_time=round(prompt_preparation_time, 3),
        llm_processing_time=round(llm_processing_time, 3),
        response_processing_time=round(response_processing_time, 3),
        api_processing_time=0,
        total_processing_time=0,
        tokens_per_second=(
            round(tokens_per_second, 2) if tokens_per_second is not None else None
        ),
        estimated_cost_usd=estimated_cost,
    )
