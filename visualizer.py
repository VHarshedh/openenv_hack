"""
visualizer.py

Streamlit dashboard to replay agent trajectories from inference.py JSON logs.

Schema (inference.py): {
  "model": str,
  "timestamp": str,
  "tasks": [
    {
      "task_idx": int,
      "difficulty": str | null,
      "final_reward": float,   # primary; visualizer also accepts reward / score / last-step reward
      "steps": [
        {"tool_name", "arguments", "result", "reward", "done"} |
        {"action": "invalid_json", "error", ...} |
        {"action": "chat", "content"}
      ]
    }
  ]
}

Usage:
    pip install streamlit pandas
    streamlit run visualizer.py
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st


def _safe_float(value: Any, default: float = 0.01) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


def _task_run_score(run: dict[str, Any]) -> float:
    """Match inference.py: primary key is `final_reward`; keep legacy `reward` / `score`."""
    for key in ("final_reward", "reward", "score", "finalScore", "task_score"):
        if key not in run:
            continue
        raw = run.get(key)
        if raw is None:
            continue
        return _safe_float(raw, default=0.01)
    # Fallback: last step reward on a terminal step (older or partial exports)
    steps = run.get("steps") or run.get("history") or []
    if isinstance(steps, list):
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            if step.get("done") is True and "reward" in step:
                return _safe_float(step.get("reward"), default=0.01)
        if steps:
            last = steps[-1]
            if isinstance(last, dict) and last.get("reward") is not None:
                return _safe_float(last.get("reward"), default=0.01)
    return 0.01


def _load_runs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        runs = data.get("tasks") or data.get("runs") or []
        if not runs and isinstance(data.get("results"), dict):
            inner = data["results"]
            runs = inner.get("tasks") or inner.get("runs") or []
        return [r for r in runs if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


st.set_page_config(page_title="ComplianceGuard Replay", page_icon="🛡️", layout="wide")

st.title("🛡️ ComplianceGuard: Visual Replay Dashboard")
st.markdown(
    "Upload a JSON file from **`results/`** (written by `inference.py`) or any compatible run log."
)

uploaded_file = st.file_uploader("Upload run JSON", type=["json"])

if uploaded_file is not None:
    try:
        raw = uploaded_file.read()
        data = json.loads(raw.decode("utf-8"))
        runs = _load_runs(data)

        if not runs:
            st.error(
                "No task trajectories found. Expected top-level **`tasks`** (inference.py format) "
                "or **`runs`** / **`results.tasks`**."
            )
            st.stop()

        st.sidebar.header("Select Task")

        task_options = []
        for i, run in enumerate(runs):
            score = _task_run_score(run)
            diff = run.get("difficulty")
            suffix = f" [{diff}]" if diff else ""
            task_options.append(f"Task {i + 1}{suffix} (score {score:.2f})")

        selected_task_str = st.sidebar.selectbox("Choose trajectory:", task_options)
        selected_idx = task_options.index(selected_task_str)
        selected_run = runs[selected_idx]

        history_raw = selected_run.get("steps") or selected_run.get("history") or []
        history = [s for s in history_raw if isinstance(s, dict)]

        score = _task_run_score(selected_run)

        if score >= 0.8:
            st.success(f"### Final reward: {score:.2f} — SOP compliant")
        else:
            st.error(f"### Final reward: {score:.2f} — Below target")

        st.markdown("---")
        st.subheader("Agent timeline")

        for step_idx, step in enumerate(history):
            tool_name = step.get("tool_name") or step.get("action") or "unknown"
            act = step.get("action")

            if act == "invalid_json":
                args = {}
                thought = str(step.get("error", "Invalid JSON from model."))
            elif act == "chat":
                args = {}
                thought = str(step.get("content", "No text."))
            else:
                raw_args = step.get("arguments")
                if isinstance(raw_args, dict):
                    args = {k: v for k, v in raw_args.items() if k != "thought"}
                    thought = str(raw_args.get("thought", "No thought provided."))
                else:
                    args = {}
                    thought = "No thought provided."

            res_str = str(step.get("result", step.get("error", "")))
            step_reward = _safe_float(step.get("reward"), default=0.01)
            done_step = step.get("done")

            title = f"Step {step_idx + 1}: {tool_name}"
            if done_step is not None:
                title += f" · r={step_reward:.2f} · done={done_step}"

            with st.expander(title, expanded=step_idx < 2):
                st.markdown(f"**Thought**\n> {thought}")
                try:
                    st.markdown(f"**Arguments** `{json.dumps(args, default=str)}`")
                except TypeError:
                    st.markdown(f"**Arguments** `{args!r}`")

                if "SYSTEM_REJECT" in res_str:
                    st.error(f"**Environment block**\n\n{res_str}")
                elif "SYSTEM_WARNING" in res_str:
                    st.warning(f"**Environment warning**\n\n{res_str}")
                elif res_str:
                    st.info(f"**Observation**\n\n{res_str}")
                else:
                    st.caption("No observation text for this step.")

        if history:
            try:
                df = pd.DataFrame(history)
                st.subheader("Raw step table")
                st.dataframe(df, use_container_width=True)
            except Exception:
                pass

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload a JSON log to begin.")

    st.markdown("### Why this matters")
    st.markdown(
        """
        This dashboard highlights **process supervision**: soft-blocks, warnings, and per-step rewards
        from the support triage environment — aligned with `inference.py` run exports.
        """
    )
