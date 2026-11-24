from typing import Any, Optional, Tuple

import requests
import streamlit as st


def fetch_runtime_metrics(api_url: str) -> Tuple[Optional[dict[str, Any]], str]:
    try:
        response = requests.get(f"{api_url}/api/metrics", timeout=5)
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        return None, f"Interview API returned a metrics error: {detail}"
    except requests.RequestException as error:
        return None, f"Runtime metrics are unavailable: {error}"
    return response.json(), ""


def render_current_metrics(
    placeholder: Any, process_info: Optional[dict[str, Any]]
) -> None:
    with placeholder.container():
        st.markdown("### Current Metrics")
        if not process_info:
            st.caption("Current request metrics appear after question generation.")
            return

        st.caption(f"Model: {process_info['model']}")
        if process_info.get("model_selection"):
            st.caption(f"🤖 {process_info['model_selection']}")
        from_history = process_info["model"] == "Own knowledge base (saved history)"
        if from_history:
            st.info("Retrieved from saved history; no LLM call was made.")
        else:
            st.metric(
                "Total tokens",
                (
                    process_info["token_size"]
                    if process_info["token_size"] is not None
                    else "Unavailable"
                ),
            )
            st.caption(
                f"Input: {process_info['input_tokens'] if process_info['input_tokens'] is not None else 'Unavailable'} · "
                f"Output: {process_info['output_tokens'] if process_info['output_tokens'] is not None else 'Unavailable'}"
            )
        st.metric(
            "Total API processing time",
            f"{process_info['total_processing_time']:.2f} s",
        )
        if not from_history:
            st.caption(
                f"LLM call and structured parsing: "
                f"{process_info['llm_processing_time']:.2f} s"
            )
        st.caption(
            f"Application/API processing (non-LLM): "
            f"{process_info['api_processing_time']:.2f} s"
        )
        with st.expander("Processing time breakdown"):
            st.caption(
                f"Request validation and source selection: "
                f"{process_info['request_validation_time']:.3f} s"
            )
            st.caption(f"Model setup: {process_info['model_setup_time']:.3f} s")
            st.caption(
                f"Prompt preparation: "
                f"{process_info['prompt_preparation_time']:.3f} s"
            )
            st.caption(
                f"LLM call and structured parsing: "
                f"{process_info['llm_processing_time']:.3f} s"
            )
            st.caption(
                f"Response validation and assembly: "
                f"{process_info['response_processing_time']:.3f} s"
            )
            st.caption(
                "Total covers API-handler time through response assembly. "
                "Network transfer and machine-wide OS/system time are not measured."
            )
        if not from_history and process_info["tokens_per_second"] is not None:
            st.caption(
                f"Output speed: {process_info['tokens_per_second']:.1f} tokens/s"
            )
        if not from_history and process_info["estimated_cost_usd"] is not None:
            st.caption(
                f"Estimated provider cost: "
                f"${process_info['estimated_cost_usd']:.6f}"
            )
        st.caption("Per-request metrics; not a scalability/load-test result.")


def render_overall_metrics(
    placeholder: Any,
    metrics: Optional[dict[str, Any]],
    error: str,
) -> None:
    with placeholder.container():
        st.markdown("### Overall Metrics")
        if error:
            st.warning(error)
            return
        if not metrics:
            st.caption("Overall runtime metrics are not available yet.")
            return

        st.metric(
            "Requests since start",
            metrics["request_count_since_start"],
        )
        st.metric(
            "Server errors",
            f"{metrics['server_error_count_since_start']} "
            f"({metrics['server_error_rate_percent_since_start']:.1f}%)",
        )
        st.metric(
            "Current in-flight",
            metrics["in_flight_requests"],
        )
        st.metric(
            "Peak concurrent observed",
            metrics["max_concurrent_requests_observed"],
        )
        st.metric(
            "Average / p95 latency",
            f"{metrics['average_latency_seconds_last_sample']:.2f} / "
            f"{metrics['p95_latency_seconds_last_sample']:.2f} s",
        )
        st.metric(
            "Observed throughput",
            f"{metrics['observed_requests_per_second_last_60_seconds']:.2f} req/s",
            help=(
                f"{metrics['requests_last_60_seconds']} requests in the "
                "last 60 seconds."
            ),
        )
        st.caption(
            f"Uptime: {metrics['uptime_seconds']:.0f} s · "
            f"Latency sample: {metrics['completed_latency_sample_size']} "
            f"of the latest {metrics['latency_sample_limit']} completed requests."
        )
        process_runs = [
            (key[: -len("_run_count")], value)
            for key, value in metrics.items()
            if key.startswith("process_") and key.endswith("_run_count")
        ]
        if process_runs:
            with st.expander("Per-process monitoring"):
                st.write(
                    [
                        {
                            "Process": prefix.removeprefix("process_")
                            .replace("_", " ")
                            .title(),
                            "Runs": run_count,
                            "Succeeded": metrics[f"{prefix}_success_count"],
                            "Failed": metrics[f"{prefix}_failure_count"],
                            "Average (s)": metrics[
                                f"{prefix}_average_elapsed_seconds"
                            ],
                            "Last (s)": metrics[f"{prefix}_last_elapsed_seconds"],
                            "Max (s)": metrics[f"{prefix}_max_elapsed_seconds"],
                        }
                        for prefix, run_count in process_runs
                    ]
                )
        st.caption(metrics["scope"])
        st.caption(metrics["scalability_note"])
