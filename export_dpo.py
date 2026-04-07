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
from pathlib import Path

def format_trajectory_as_text(history: list) -> str:
    """Converts a step-by-step history array into a readable LLM string."""
    text_blocks = []
    for step in history:
        # Format the agent's action
        if "action" in step:
            tool = step["action"].get("tool_name", "unknown_tool")
            args = step["action"].get("arguments", {})
            thought = args.pop("thought", "")
            
            action_text = f"Thought: {thought}\nAction: {tool}\nArguments: {json.dumps(args)}"
            text_blocks.append(action_text)
            
        # Format the environment's observation
        if "observation" in step:
            obs = step["observation"]
            res = obs.get("result", "")
            if isinstance(res, dict):
                res = res.get("data", res.get("content", str(res)))
            
            text_blocks.append(f"Observation: {res}")
            
    return "\n\n".join(text_blocks)

def main():
    parser = argparse.ArgumentParser(description="Export DPO Dataset from OpenEnv Logs")
    parser.add_argument("--log_dir", type=str, default="./results", help="Directory containing results.json")
    parser.add_argument("--out_file", type=str, default="dpo_dataset.jsonl", help="Output file name")
    args = parser.parse_args()

    log_path = Path(args.log_dir) / "results.json"
    if not log_path.exists():
        print(f"No results.json found in {args.log_dir}. Please run inference.py first.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error parsing {log_path}. Ensure it is valid JSON.")
            return

    # In OpenEnv, results.json is typically a dict mapping task/eval IDs to results
    runs = data.get("runs", data) if isinstance(data, dict) else data
    if isinstance(runs, dict):
         runs = list(runs.values())

    dpo_pairs = []
    
    # We need to find matching prompts to pair chosen and rejected trajectories.
    # Group by task difficulty/metadata.
    task_groups = {}
    
    for run in runs:
        score = run.get("reward", run.get("score", 0.0))
        history = run.get("history", [])
        
        if not history:
            continue
            
        # Extract the initial ticket/prompt (usually the first read_ticket result)
        initial_prompt = "Customer Support Ticket"
        for step in history:
            if "action" in step and step["action"].get("tool_name") == "read_ticket":
                obs = step.get("observation", {}).get("result", "")
                initial_prompt = str(obs).split("[METADATA")[0].strip()
                break
                
        trajectory_text = format_trajectory_as_text(history)
        
        if initial_prompt not in task_groups:
            task_groups[initial_prompt] = {"chosen": None, "rejected": None}
            
        # Assign Chosen
        if score >= 0.8:
            task_groups[initial_prompt]["chosen"] = trajectory_text
            
        # Assign Rejected (Checking for our custom Hard-Stops)
        if "SYSTEM_REJECT" in trajectory_text or score < 0.3:
            task_groups[initial_prompt]["rejected"] = trajectory_text

    # Compile valid DPO pairs
    for prompt, group in task_groups.items():
        if group["chosen"] and group["rejected"]:
            dpo_pairs.append({
                "system": "You are a strict enterprise support agent. You must adhere to SOPs and avoid hallucinating.",
                "prompt": prompt,
                "chosen": group["chosen"],
                "rejected": group["rejected"]
            })

    with open(args.out_file, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[OK] Successfully exported {len(dpo_pairs)} DPO training pairs to {args.out_file}!")
    print("This dataset is ready to be uploaded to Hugging Face for fine-tuning Llama 4 or Qwen models.")

if __name__ == "__main__":
    main()