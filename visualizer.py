"""
visualizer.py

A Streamlit dashboard to visually replay agent trajectories.
Highlights SYSTEM_REJECT soft-blocks to showcase process supervision.

Usage:
    pip install streamlit pandas
    streamlit run visualizer.py
"""
import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="ComplianceGuard Replay", page_icon="🛡️", layout="wide")

st.title("🛡️ ComplianceGuard: Visual Replay Dashboard")
st.markdown("Upload a `results.json` file from an evaluation run to view step-by-step agent trajectories and SOP enforcement.")

uploaded_file = st.file_uploader("Upload results.json", type=["json"])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        # inference.py writes: { "tasks": [ { "steps": [...], "final_reward", ... }, ... ] }
        # Legacy / alt layouts: top-level "runs", or raw list, or nested under "results".
        runs: list = []
        if isinstance(data, dict):
            runs = data.get("tasks") or data.get("runs") or []
            if not runs and isinstance(data.get("results"), dict):
                inner = data["results"]
                runs = inner.get("tasks") or inner.get("runs") or []
        elif isinstance(data, list):
            runs = data

        if not runs:
            st.error(
                "Could not find task trajectories. Expected a JSON object with a **tasks** array "
                "(as produced by inference.py), or legacy **runs**."
            )
            st.stop()
            
        st.sidebar.header("Select Task")
        
        # Create a dropdown for tasks
        task_options = []
        for i, run in enumerate(runs):
            # Support both schema formats for final score
            score = run.get("final_reward", run.get("reward", run.get("score", 0.0)))
            task_options.append(f"Task {i+1} (Score: {score:.2f})")
            
        selected_task_str = st.sidebar.selectbox("Choose Trajectory:", task_options)
        selected_idx = task_options.index(selected_task_str)
        
        selected_run = runs[selected_idx]

        # Step timeline: inference.py uses "steps"; older files used "history".
        history = (
            selected_run.get("steps")
            or selected_run.get("history")
            or []
        )
        score = selected_run.get("final_reward", selected_run.get("reward", 0.0))
        
        # Determine status
        if score >= 0.8:
            st.success(f"### Final Reward: {score:.2f} - SOP Compliant ✅")
        else:
            st.error(f"### Final Reward: {score:.2f} - SOP Violation ❌")
            
        st.markdown("---")
        
        # Replay the timeline
        st.subheader("Agent Timeline")
        
        for step_idx, step in enumerate(history):
            # inference.py: tool_name + arguments; chat fallbacks use "action" / "content"
            tool_name = step.get("tool_name") or step.get("action") or "unknown"
            raw_args = step.get("arguments")
            if isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {}
            thought = args.pop("thought", "No thought provided.")
            
            # Extract raw result text
            res_str = str(step.get("result", ""))
            
            with st.expander(f"Step {step_idx + 1}: {tool_name}", expanded=True):
                st.markdown(f"**🧠 Thought:**\n> {thought}")
                st.markdown(f"**🔧 Arguments:** `{json.dumps(args)}`")
                
                # Critical Highlight Feature
                if "SYSTEM_REJECT" in res_str:
                    st.error(f"**🚫 ENVIRONMENT BLOCK:**\n\n{res_str}")
                elif "SYSTEM_WARNING" in res_str:
                    st.warning(f"**⚠️ ENVIRONMENT WARNING:**\n\n{res_str}")
                else:
                    st.info(f"**🔍 Observation:**\n\n{res_str}")
                    
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
else:
    st.info("Awaiting file upload...")
    
    # Show placeholder data feature to judges
    st.markdown("### Why this matters")
    st.markdown("""
    This dashboard demonstrates our **Process Supervision** architecture. 
    Unlike standard benchmarks that only check final answers, ComplianceGuard provides:
    * **Step-by-step RLHF rewards**
    * **Strict Soft-Blocks** (Highlighted in red) when agents hallucinate or skip required SOP steps.
    * **Dynamic telemetry** for generating fine-tuning datasets.
    """)