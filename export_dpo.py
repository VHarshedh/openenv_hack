"""
export_dpo.py

This script converts OpenEnv evaluation logs into a Hugging Face DPO 
(Direct Preference Optimization) dataset format (.jsonl).

It acts as a Data Engine:
- "Chosen" responses are trajectories that scored >= 0.8.
- "Rejected" responses are trajectories where the agent hit a SYSTEM_REJECT soft-block.

Usage:
    python export_dpo.py --log_dir ./results
"""
import json
import os
import argparse
import glob
from pathlib import Path

def format_trajectory_as_text(history: list) -> str:
    """Converts a step-by-step history array into a readable LLM string."""
    text_blocks = []
    for step in history:
        # Format the agent's action
        tool = step.get("action", {}).get("tool_name") or step.get("tool_name")
        args = step.get("action", {}).get("arguments") or step.get("arguments", {})
        
        if tool is not None:
            # Avoid mutating the original dictionary inside the loop
            args_copy = dict(args) if isinstance(args, dict) else {}
            thought = args_copy.pop("thought", "")
            action_text = f"Thought: {thought}\nAction: {tool}\nArguments: {json.dumps(args_copy)}"
            text_blocks.append(action_text)
            
        # Format the environment's observation
        if "observation" in step:
            res = step["observation"].get("result", "")
        else:
            res = step.get("result", "")
            
        if isinstance(res, dict):
            res = res.get("data", res.get("content", str(res)))
            
        if "observation" in step or "result" in step:
            text_blocks.append(f"Observation: {res}")
            
    return "\n\n".join(text_blocks)

def main():
    parser = argparse.ArgumentParser(description="Export DPO Dataset from OpenEnv Logs")
    parser.add_argument("--log_dir", type=str, default="./results", help="Directory containing results JSON files")
    parser.add_argument("--out_file", type=str, default="dpo_dataset.jsonl", help="Output file name")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Directory {args.log_dir} does not exist.")
        return

    json_files = glob.glob(str(log_dir / "*.json"))
    if not json_files:
        print(f"No .json files found in {args.log_dir}. Please run inference.py first.")
        return

    runs = []
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                file_runs = data.get("runs", data.get("tasks", data))
                if isinstance(file_runs, dict):
                    file_runs = list(file_runs.values())
                if isinstance(file_runs, list):
                    runs.extend(file_runs)
            except json.JSONDecodeError:
                print(f"Error parsing {filepath}. Skipping.")

    dpo_pairs = []
    
    # We need to find matching prompts to pair chosen and rejected trajectories.
    # Group by task difficulty/metadata.
    task_groups = {}
    
    for run in runs:
        score = run.get("final_reward", run.get("reward", run.get("score", 0.0)))
        history = run.get("history", run.get("steps", []))
        
        if not history:
            continue
            
        # Extract the initial ticket/prompt (usually the first read_ticket result)
        initial_prompt = "Customer Support Ticket"
        for step in history:
            tool = step.get("action", {}).get("tool_name") or step.get("tool_name")
            if tool == "read_ticket":
                if "observation" in step:
                    obs = step["observation"].get("result", "")
                else:
                    obs = step.get("result", "")
                initial_prompt = str(obs).split("[METADATA")[0].strip()
                break
                
        trajectory_text = format_trajectory_as_text(history)
        
        if initial_prompt not in task_groups:
            task_groups[initial_prompt] = {"chosen": [], "rejected": []}
            
        # Assign Chosen
        if score >= 0.8:
            task_groups[initial_prompt]["chosen"].append(trajectory_text)
            
        # Assign Rejected (Checking for our custom Hard-Stops)
        if "SYSTEM_REJECT" in trajectory_text or score <= 0.3:
            task_groups[initial_prompt]["rejected"].append(trajectory_text)

    # Compile valid DPO pairs by matching chosen to rejected
    for prompt, group in task_groups.items():
        # Cartesian product or simple 1:1 pairing
        if group["chosen"] and group["rejected"]:
            for chosen_traj in group["chosen"]:
                for rejected_traj in group["rejected"]:
                    dpo_pairs.append({
                        "system": "You are a strict enterprise support agent. You must adhere to SOPs and avoid hallucinating.",
                        "prompt": prompt,
                        "chosen": chosen_traj,
                        "rejected": rejected_traj
                    })

    with open(args.out_file, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[OK] Successfully exported {len(dpo_pairs)} DPO training pairs to {args.out_file}!")
    print("This dataset is ready to be uploaded to Hugging Face for fine-tuning Llama 4 or Qwen models.")

if __name__ == "__main__":
    main()