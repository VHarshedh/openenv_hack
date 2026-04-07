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
# Replace the strict .env check with this graceful load:
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH, override=True)

# os.getenv() picks up evaluator-injected env vars; optional `.env` only augments local dev.
def log_info(msg: str):
    """Utility to print non-essential logs to stderr so stdout remains pure for the grader."""
    print(msg, file=sys.stderr)


def _strip(s: str | None) -> str:
    return (s or "").strip()


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        v = _strip(os.getenv(name))
        if v:
            return v
    return ""


# Rubric / sample script: HF_TOKEN or OPENAI_API_KEY (or API_KEY). No `.env` required in containers.
API_KEY = _first_nonempty_env("HF_TOKEN", "OPENAI_API_KEY", "API_KEY")
if not API_KEY:
    log_info(
        "❌ ERROR: Set one of HF_TOKEN, OPENAI_API_KEY, or API_KEY in the environment "
        "(evaluators may inject only OPENAI_API_KEY)."
    )
    sys.exit(1)
if "PASTE" in API_KEY:
    log_info("❌ ERROR: Replace placeholder API key in the environment.")
    sys.exit(1)

# HF Inference API router default matches hackathon sample; override with API_BASE_URL or OPENAI_BASE_URL.
_DEFAULT_LLM_BASE = "https://router.huggingface.co/v1"
API_BASE_URL = _first_nonempty_env("API_BASE_URL", "OPENAI_BASE_URL") or _DEFAULT_LLM_BASE

# Environment server (OpenEnv HTTP) — sidecar on Spaces is usually localhost:8000.
ENV_URL = _strip(os.getenv("ENV_URL", "http://127.0.0.1:8000")).rstrip("/")
MODEL_NAME = _strip(os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))

# ---------------------------------------------------------------------------
# Configuration (defaults below are optional tuning only)
# ---------------------------------------------------------------------------

NUM_TASKS = int(os.getenv("NUM_TASKS", "10"))
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "15"))

# Wall-clock budget (default 20 min). Raise only if evaluators allow (e.g. INFERENCE_MAX_SECONDS=1800).
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1200"))

# Phase 2: task scores must satisfy 0 < score < 1 (not 0.0, not 1.0). Also avoid printing
# "1.00" / "0.00" from rounding (e.g. 0.997 → 1.00 breaks strict parsers).
_PHASE2_MIN = 0.01
_PHASE2_MAX = 0.99


def _clamp_phase2_score(raw) -> float:
    try:
        if raw is None:
            x = _PHASE2_MIN
        else:
            x = float(raw)
    except (TypeError, ValueError):
        x = _PHASE2_MIN
    if x != x:  # NaN
        x = _PHASE2_MIN
    return max(_PHASE2_MIN, min(_PHASE2_MAX, x))


def _fmt_phase2_reward(x: float) -> str:
    return f"{_clamp_phase2_score(x):.2f}"


def _configured_step_delay_seconds() -> int:
    """Artificial pause between steps (after step 1). Default 0 so baseline finishes within 20 min.

    Set STEP_DELAY_SECONDS>0 locally if your provider rate-limits aggressive calls.
    """
    override = os.getenv("STEP_DELAY_SECONDS")
    if override is not None and override.strip() != "":
        return max(0, int(override))
    # Evaluators typically do not set this — avoid multi-minute runs from 7s/30s sleeps.
    return 0


def _effective_step_delay(
    base_delay: int,
    *,
    run_start: float,
    task_idx: int,
) -> int:
    """Drop artificial delay when wall-clock budget is tight so all tasks can finish."""
    if base_delay <= 0:
        return 0
    elapsed = time.time() - run_start
    remaining = INFERENCE_MAX_SECONDS - elapsed
    tasks_left = max(1, NUM_TASKS - task_idx)
    # Reserve time for ~25s LLM+env per potential step (conservative); skip sleep if under water.
    reserve = tasks_left * MAX_STEPS_PER_TASK * 25 + 90
    if remaining < reserve:
        return 0
    return base_delay


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
    step_delay = _configured_step_delay_seconds()

    def time_elapsed() -> float:
        return time.time() - run_start

    def time_budget_exceeded() -> bool:
        if time_elapsed() >= INFERENCE_MAX_SECONDS:
            log_info(
                f"❌ Stopping: INFERENCE_MAX_SECONDS ({INFERENCE_MAX_SECONDS}s) exceeded. "
                "Increase INFERENCE_MAX_SECONDS if your platform allows, or reduce NUM_TASKS / MAX_STEPS_PER_TASK."
            )
            return True
        return False

    log_info("⏳ Waiting for environment server to start...")
    _server_ok = False

    # Use httpx.AsyncClient for non-blocking HTTP requests
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for attempt in range(20):
            try:
                for path in ("/health", "/docs"):
                    r = await http_client.get(f"{ENV_URL}{path}", timeout=2.0)
                    if r.status_code == 200:
                        _server_ok = True
                        break
                if _server_ok:
                    break
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)

        if not _server_ok:
            log_info("❌ ERROR: Environment server is not reachable after 20 retries (~20s).")
            sys.exit(1)

        log_info("✅ Environment server is reachable!")
        log_info(f"🚀 Starting inference with {MODEL_NAME}...")
        log_info(
            f"⏳ Artificial step delay: {step_delay}s (STEP_DELAY_SECONDS; default 0 for eval). "
            f"Wall budget: {INFERENCE_MAX_SECONDS}s. LLM: {API_BASE_URL}\n"
        )

        client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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
            
            # FIX: Explicitly fetch the state to get the episode_id required for Phase 1 Concurrency
            state_resp = await http_client.get(f"{ENV_URL}/state", timeout=10.0)
            state_resp.raise_for_status()
            episode_id = state_resp.json().get("episode_id")

            task_log = {
                "task_idx": task_idx,
                "difficulty": response.json().get("observation", {}).get("metadata", {}).get("difficulty"),
                "steps": [],
                "final_reward": 0.01
            }

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "New ticket arrived. Begin."},
            ]

            done = False
            step_count = 0
            final_reward = 0.01
            rewards_history = []

            try:
                while not done and step_count < MAX_STEPS_PER_TASK:
                    if time_budget_exceeded():
                        break

                    step_count += 1

                    if step_count > 1:
                        delay = _effective_step_delay(
                            step_delay, run_start=run_start, task_idx=task_idx
                        )
                        if delay > 0:
                            log_info(f"   ⏳ Waiting {delay}s before next step...")
                            await asyncio.sleep(delay)

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
                                cool = int(os.getenv("RATE_LIMIT_SLEEP_SECONDS", "20"))
                                cool = max(5, min(60, cool))
                                log_info(f"   🚨 Rate limit/Server busy. Cooling down for {cool}s...")
                                await asyncio.sleep(cool)
                            else:
                                break

                    if not api_success:
                        log_info("   ❌ API failed after retries. Aborting task.")
                        # STDOUT failure step log before breaking
                        print(f"[STEP] step={step_count} action=api_fail() reward=0.01 done=true error=\"API failed\"", flush=True)
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
                            
                            rewards_history.append(0.01)
                            print(f"[STEP] step={step_count} action={name}(invalid_json) reward=0.01 done=false error=\"invalid_json\"", flush=True)
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
                        reward = _clamp_phase2_score(res.get("reward"))

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
                        print(
                            f"[STEP] step={step_count} action={action_str} reward={_fmt_phase2_reward(reward)} done={done_str} error=null",
                            flush=True,
                        )

                        if done:
                            final_reward = reward
                            log_info(f"   ✅ Complete! Final Reward: {round(final_reward, 2)}")
                            break
                    else:
                        safe_content = str(response_message.content or "No text provided.")
                        log_info(f"   🤖 Agent chatted: {safe_content[:50]}...")
                        messages.append({"role": "user", "content": "Focus. Use a tool to progress."})
                        task_log["steps"].append({"action": "chat", "content": safe_content})
                        
                        rewards_history.append(0.01)
                        # >>> STDOUT MANDATORY REQUIREMENT: [STEP] (for non-tool responses) <<<
                        print(f"[STEP] step={step_count} action=chat() reward=0.01 done=false error=\"Did not call tool\"", flush=True)

            finally:
                # >>> STDOUT MANDATORY REQUIREMENT: [END] <<<
                # (Executes even if exception occurs to ensure compliance)
                final_reward = _clamp_phase2_score(final_reward)
                success_str = "true" if done and final_reward > 0.1 else "false"
                if not rewards_history:
                    rewards_str = _fmt_phase2_reward(_PHASE2_MIN)
                else:
                    rewards_str = ",".join(_fmt_phase2_reward(r) for r in rewards_history)
                # Grader regex requires score=<float> on the same line as [END].
                print(
                    f"[END] success={success_str} steps={step_count} score={_fmt_phase2_reward(final_reward)} rewards={rewards_str}",
                    flush=True,
                )

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