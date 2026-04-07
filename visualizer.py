"""
visualizer.py

A Streamlit dashboard to visually replay agent trajectories.
Highlights SYSTEM_REJECT soft-blocks to showcase process supervision.

Usage:
    pip install streamlit
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
        runs = data.get("runs", data) if isinstance(data, dict) else data
        if isinstance(runs, dict):
            runs = list(runs.values())
            
        st.sidebar.header("Select Task")
        
        # Create a dropdown for tasks
        task_options = []
        for i, run in enumerate(runs):
            score = run.get("reward", run.get("score", 0.0))
            task_options.append(f"Task {i+1} (Score: {score:.2f})")
            
        selected_task_str = st.sidebar.selectbox("Choose Trajectory:", task_options)
        selected_idx = task_options.index(selected_task_str)
        
        selected_run = runs[selected_idx]
        history = selected_run.get("history", [])
        score = selected_run.get("reward", selected_run.get("score", 0.0))
        
        # Determine status
        if score >= 0.8:
            st.success(f"### Final Reward: {score:.2f} - SOP Compliant ✅")
        else:
            st.error(f"### Final Reward: {score:.2f} - SOP Violation ❌")
            
        st.markdown("---")
        
        # Replay the timeline
        st.subheader("Agent Timeline")
        
        for step_idx, step in enumerate(history):
            action = step.get("action", {})
            obs = step.get("observation", {})
            
            tool_name = action.get("tool_name", "unknown")
            args = action.get("arguments", {})
            thought = args.pop("thought", "No thought provided.")
            
            # Extract raw result text
            res = obs.get("result", "")
            if isinstance(res, dict):
                res = res.get("data", res.get("content", str(res)))
            res_str = str(res)
            
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