#!/usr/bin/env python3
import asyncio
import os
import json
import httpx
import sys
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Only the real `.env` next to this file (not cwd, not `.env.example`).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if not _ENV_PATH.is_file():
    print(f"❌ ERROR: Missing {_ENV_PATH}", file=sys.stderr)
    print("   Create that file and set ENV_URL, API_BASE_URL, MODEL_NAME, HF_TOKEN.", file=sys.stderr)
    sys.exit(1)
load_dotenv(_ENV_PATH, override=True)


def log_info(msg: str):
    """Utility to print non-essential logs to stderr so stdout remains pure for the grader."""
    print(msg, file=sys.stderr)


def _strip(s: str | None) -> str:
    return (s or "").strip()


_REQUIRED = ("HF_TOKEN", "ENV_URL", "API_BASE_URL", "MODEL_NAME")
_missing = [k for k in _REQUIRED if not _strip(os.getenv(k))]
if _missing:
    log_info(f"❌ ERROR: {_ENV_PATH} is missing keys: {', '.join(_missing)}")
    sys.exit(1)

HF_TOKEN = _strip(os.getenv("HF_TOKEN"))
if "PASTE" in HF_TOKEN:
    log_info("❌ ERROR: Replace placeholder HF_TOKEN in .env.")
    sys.exit(1)

ENV_URL = _strip(os.getenv("ENV_URL")).rstrip("/")
API_BASE_URL = _strip(os.getenv("API_BASE_URL"))
MODEL_NAME = _strip(os.getenv("MODEL_NAME"))

# ---------------------------------------------------------------------------
# Configuration (defaults below are optional tuning only)
# ---------------------------------------------------------------------------

NUM_TASKS = 6
MAX_STEPS_PER_TASK = 15

# Wall-clock budget for the whole run (default 20 minutes for hackathon validators)
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1200"))


def _step_delay_seconds() -> int:
    """Delay between steps (after step 1). Override with STEP_DELAY_SECONDS; else model-based default."""
    override = os.getenv("STEP_DELAY_SECONDS")
    if override is not None and override.strip() != "":
        return max(0, int(override))
    return 30 if "pro" in MODEL_NAME.lower() else 7


SYSTEM_PROMPT = """You are an expert customer support triage agent.
Respond ONLY with a tool call. Do not explain your reasoning outside of the tool call. 

TOOLS:
- `read_ticket(thought)`: Read the ticket. (ALWAYS START HERE)
- `search_knowledge_base(thought, query)`: Search policies.
- `check_billing(thought, user_id)`: Check user billing and transactions.
- `ping_human_manager(thought, reason)`: Ask a manager for help if the instructions are unclear.
- `escalate_ticket(thought, department)`: Escalate to billing, engineering, or security.
- `resolve_ticket(thought, message)`: Send the final message to the customer.

CRITICAL INSTRUCTIONS:
Every tool requires a `thought` parameter. You MUST use this parameter to think step-by-step BEFORE executing the action. 
1. Identify all distinct requests in the user's message (e.g., refund AND deletion).
2. Note any specific states found in the billing database (e.g., 'suspended').
3. Cross-reference these findings against the Knowledge Base to check for conflicting rules or required escalation paths.
4. Do not call `resolve_ticket` or `escalate_ticket` until you have verified your logic against the KB.
"""


async def main():
    run_start = time.time()
    step_delay = _step_delay_seconds()

    def time_elapsed() -> float:
        return time.time() - run_start

    def time_budget_exceeded() -> bool:
        if time_elapsed() >= INFERENCE_MAX_SECONDS:
            log_info(
                f"❌ Stopping: INFERENCE_MAX_SECONDS ({INFERENCE_MAX_SECONDS}s) exceeded. "
                "Set STEP_DELAY_SECONDS=0 or raise INFERENCE_MAX_SECONDS if needed."
            )
            return True
        return False

    log_info("⏳ Waiting for environment server to start...")
    _server_ok = False
    
    # Use httpx.AsyncClient for non-blocking HTTP requests
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for _ in range(10):
            try:
                await http_client.get(f"{ENV_URL}/docs", timeout=2.0)
                _server_ok = True
                break
            except httpx.RequestError:
                await asyncio.sleep(2)
                
        if not _server_ok:
            log_info("❌ ERROR: Environment server is not reachable after 10 retries!")
            sys.exit(1)
            
        log_info("✅ Environment server is reachable!")
        log_info(f"🚀 Starting inference with {MODEL_NAME}...")
        log_info(
            f"⏳ Step delay: {step_delay}s between steps (set STEP_DELAY_SECONDS to override). "
            f"Wall budget: {INFERENCE_MAX_SECONDS}s (INFERENCE_MAX_SECONDS).\n"
        )

        # Use AsyncOpenAI instead of OpenAI
        client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

        openai_tools = [
            {"type": "function", "function": {"name": "read_ticket", "description": "Read ticket.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]}}},
            {"type": "function", "function": {"name": "search_knowledge_base", "description": "Search policies.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "query": {"type": "string"}}, "required": ["thought", "query"]}}},
            {"type": "function", "function": {"name": "check_billing", "description": "Check billing.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "user_id": {"type": "string"}}, "required": ["thought", "user_id"]}}},
            {"type": "function", "function": {"name": "ping_human_manager", "description": "Ask a manager for help.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "reason": {"type": "string"}}, "required": ["thought", "reason"]}}},
            {"type": "function", "function": {"name": "escalate_ticket", "description": "Escalate ticket.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "department": {"type": "string"}}, "required": ["thought", "department"]}}},
            {"type": "function", "function": {"name": "resolve_ticket", "description": "Resolve ticket.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "message": {"type": "string"}}, "required": ["thought", "message"]}}},
        ]

        total_rewards = []
        run_logs = {
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "tasks": []
        }

        for task_idx in range(NUM_TASKS):
            if time_budget_exceeded():
                break

            log_info(f"\n{'='*60}\n   TASK {task_idx + 1}\n{'='*60}")

            # >>> STDOUT MANDATORY REQUIREMENT: [START] <<<
            task_name = f"task_{task_idx + 1}"
            print(f"[START] task={task_name} env=support_env model={MODEL_NAME}", flush=True)

            # Sync the environment to the specific task index using await
            response = await http_client.post(f"{ENV_URL}/reset", json={"task_idx": task_idx}, timeout=10.0)
            response.raise_for_status()
            
            reset_data = response.json()
            episode_id = reset_data.get("state", {}).get("episode_id") or reset_data.get("episode_id")

            task_log = {
                "task_idx": task_idx,
                "difficulty": reset_data.get("observation", {}).get("metadata", {}).get("difficulty"),
                "steps": [],
                "final_reward": 0.0
            }

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "New ticket arrived. Begin."},
            ]

            done = False
            step_count = 0
            final_reward = 0.0
            rewards_history = []

            try:
                while not done and step_count < MAX_STEPS_PER_TASK:
                    if time_budget_exceeded():
                        break

                    step_count += 1

                    if step_count > 1:
                        log_info(f"   ⏳ Waiting {step_delay}s before next step...")
                        await asyncio.sleep(step_delay)

                    log_info(f"   --- Step {step_count} ---")

                    api_success = False
                    for attempt in range(3):
                        try:
                            response = await client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                tools=openai_tools,
                                tool_choice="required" if "gemini" not in MODEL_NAME.lower() else "auto",
                                temperature=0.0,
                            )
                            api_success = True
                            break
                        except Exception as e:
                            log_info(f"   ⚠ API error on attempt {attempt + 1}: {e}")
                            if "429" in str(e) or "quota" in str(e).lower() or "503" in str(e):
                                log_info("   🚨 Rate limit/Server busy. Cooling down for 30s...")
                                await asyncio.sleep(30)
                            else:
                                break

                    if not api_success:
                        log_info("   ❌ API failed after retries. Aborting task.")
                        # STDOUT failure step log before breaking
                        print(f"[STEP] step={step_count} action=api_fail() reward=0.00 done=true error=\"API failed\"", flush=True)
                        break

                    response_message = response.choices[0].message

                    if response_message.tool_calls and len(response_message.tool_calls) > 1:
                        first_tc = response_message.tool_calls[0]
                        messages.append({
                            "role": "assistant",
                            "content": response_message.content or None,
                            "tool_calls": [{
                                "id": first_tc.id,
                                "type": "function",
                                "function": {"name": first_tc.function.name, "arguments": first_tc.function.arguments},
                            }],
                        })
                    else:
                        messages.append(response_message)

                    if response_message.tool_calls:
                        tool_call = response_message.tool_calls[0]
                        name = tool_call.function.name

                        invalid_json = False
                        try:
                            raw_args = tool_call.function.arguments or "{}"
                            args = json.loads(raw_args)
                            if args is None:
                                args = {}
                        except json.JSONDecodeError:
                            log_info(f"   ⚠ Model provided invalid JSON: {tool_call.function.arguments}")
                            invalid_json = True
                            args = {}

                        if invalid_json:
                            tool_out = "Error: Invalid JSON format. Please fix your syntax and try again."
                            log_info(f"   📋 Result: {tool_out}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": tool_out})
                            task_log["steps"].append({"action": "invalid_json", "error": tool_out})
                            
                            rewards_history.append(0.0)
                            print(f"[STEP] step={step_count} action={name}(invalid_json) reward=0.00 done=false error=\"invalid_json\"", flush=True)
                            continue

                        log_info(f"   🔧 Tool: {name}({json.dumps(args)})")

                        step_payload = {"action": {"tool_name": name, "arguments": args}}
                        if episode_id:
                            step_payload["episode_id"] = episode_id

                        step_resp = await http_client.post(
                            f"{ENV_URL}/step",
                            json=step_payload,
                        )
                        step_resp.raise_for_status()
                        res = step_resp.json()

                        res_data = res.get("observation")
                        if res_data is None:
                            tool_out = "Error: Environment returned no data."
                        else:
                            result_obj = res_data.get("result", "Success")
                            if isinstance(result_obj, dict):
                                tool_out = str(result_obj.get("data", result_obj))
                            else:
                                tool_out = str(result_obj)

                        done = res.get("done", False)
                        reward = res.get("reward", 0.0)
                        
                        log_info(f"   📋 Result: {tool_out[:100]}...")
                        log_info(f"   💰 Reward: {round(reward, 2)}")

                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": tool_out})
                        
                        task_log["steps"].append({
                            "tool_name": name,
                            "arguments": args,
                            "result": tool_out,
                            "reward": reward,
                            "done": done
                        })

                        # >>> STDOUT MANDATORY REQUIREMENT: [STEP] <<<
                        action_str = f"{name}({json.dumps(args)})"
                        # Escape any newlines in action string just to be safe
                        action_str = action_str.replace('\n', ' ').replace('\r', '')
                        done_str = "true" if done else "false"
                        rewards_history.append(reward)
                        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error=null", flush=True)

                        if done:
                            final_reward = reward
                            log_info(f"   ✅ Complete! Final Reward: {round(final_reward, 2)}")
                            break
                    else:
                        safe_content = str(response_message.content or "No text provided.")
                        log_info(f"   🤖 Agent chatted: {safe_content[:50]}...")
                        messages.append({"role": "user", "content": "Focus. Use a tool to progress."})
                        task_log["steps"].append({"action": "chat", "content": safe_content})
                        
                        rewards_history.append(0.0)
                        # >>> STDOUT MANDATORY REQUIREMENT: [STEP] (for non-tool responses) <<<
                        print(f"[STEP] step={step_count} action=chat() reward=0.00 done=false error=\"Did not call tool\"", flush=True)

            finally:
                # >>> STDOUT MANDATORY REQUIREMENT: [END] <<<
                # (Executes even if exception occurs to ensure compliance)
                success_str = "true" if done and final_reward > 0.0 else "false"
                if not rewards_history:
                    rewards_str = "0.00"
                else:
                    rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
                print(f"[END] success={success_str} steps={step_count} rewards={rewards_str}", flush=True)

            task_log["final_reward"] = final_reward
            run_logs["tasks"].append(task_log)
            total_rewards.append(final_reward)

        if not total_rewards:
            log_info("\n❌ No tasks completed.")
            sys.exit(1)

        log_info(f"\n{'='*60}\n   SUMMARY: Average Reward: {round(sum(total_rewards)/len(total_rewards), 2)}\n{'='*60}")
        
        # Save detailed logs
        os.makedirs("results", exist_ok=True)
        safe_model_name = MODEL_NAME.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"results/{safe_model_name}_run_{timestamp}.json"
        
        with open(log_file, "w") as f:
            json.dump(run_logs, f, indent=2)
        log_info(f"📁 Detailed run logs saved to {log_file}")


if __name__ == "__main__":
    asyncio.run(main())