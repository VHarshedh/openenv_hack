#!/usr/bin/env python3
import os
import json
import httpx
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv(override=True) 

# Verify API Key
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN or "PASTE" in HF_TOKEN:
    print("❌ ERROR: Your API key is missing from the .env file!")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Updated fallback in inference.py
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
NUM_TASKS = 4
MAX_STEPS_PER_TASK = 10

SYSTEM_PROMPT = """You are an expert customer support triage agent. 
Respond ONLY with a tool call. Do not explain your reasoning. Do not say 'I understand'.

TOOLS:
- `read_ticket()`: Read the ticket. (ALWAYS START HERE)
- `search_knowledge_base(query)`: Search policies.
- `check_billing(user_id)`: Check user status.
- `escalate_ticket(department)`: Use if tools don't solve it (billing, engineering, security).
- `resolve_ticket(message)`: Final answer to user.

CRITICAL RULES:
1. PASSWORD RESETS: Search the knowledge base for "password reset". Do NOT check billing. You MUST call `resolve_ticket(message="I have sent a reset link to your email. Please check your spam folder.")`.
2. REFUNDS: Search KB, then check billing using the specific USRxxx ID. If the transaction is older than 30 days, `escalate_ticket(department='billing')`.
3. OUTAGES (500 errors): Search KB for "outage", then `escalate_ticket(department='engineering')`.
4. TRAPS / UNKNOWN: `escalate_ticket(department='engineering')`.
"""

def main():
    # Wait for the environment server to wake up
    print("⏳ Waiting for environment server to start...")
    for _ in range(10):
        try:
            httpx.get(f"{ENV_URL}/docs", timeout=2.0)
            break
        except httpx.RequestError:
            time.sleep(2)
    print("✅ Environment server is reachable!")
    print(f"🚀 Starting Stabilized Inference with {MODEL_NAME}...")
    print(f"⏳ Throttling enabled (7s per step) to respect Free Tier limits.\n")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    openai_tools = [
        {"type": "function", "function": {"name": "read_ticket", "description": "Read ticket.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "search_knowledge_base", "description": "Search policies.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "check_billing", "description": "Check billing.", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
        {"type": "function", "function": {"name": "escalate_ticket", "description": "Escalate ticket.", "parameters": {"type": "object", "properties": {"department": {"type": "string"}}, "required": ["department"]}}},
        {"type": "function", "function": {"name": "resolve_ticket", "description": "Resolve ticket.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}
    ]

    total_rewards = []

    for task_num in range(1, NUM_TASKS + 1):
        print(f"\n{'='*60}\n  TASK {task_num}\n{'='*60}")
        
        # Reset Environment
        httpx.post(f"{ENV_URL}/reset", timeout=10.0)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "New ticket arrived. Begin."}
        ]

        done = False
        step_count = 0
        final_reward = 0.0

        while not done and step_count < MAX_STEPS_PER_TASK:
            step_count += 1
            
            # --- RATE LIMIT PROTECTION ---
            if step_count > 1:
                print(f"  ⏳ Waiting 7s to prevent 429 quota error...")
                time.sleep(7)

            print(f"  --- Step {step_count} ---")
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0.0 # Strictness
                )
            except Exception as e:
                print(f"  ⚠ API error: {e}")
                if "429" in str(e):
                    print("  🚨 Rate limit hit. Cooling down for 30s...")
                    time.sleep(30)
                break
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")
                    
                    print(f"  🔧 Tool: {name}({json.dumps(args)})")
                    
                    # Call Server
                    res = httpx.post(f"{ENV_URL}/step", json={"action": {"tool_name": name, "arguments": args}}, timeout=30.0).json()
                    
                    obs = res.get("observation", {})
                    done = res.get("done", False)
                    reward = res.get("reward", 0.0)
                    
                    tool_out = str(obs.get("result", {}).get("data", obs.get("result", "Success")))
                    print(f"  📋 Result: {tool_out[:100]}...")
                    print(f"  💰 Reward: {reward}")
                    
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": tool_out})
                    
                    if done:
                        final_reward = reward
                        print(f"  ✅ Complete! Final Reward: {final_reward}")
                        break
            else:
                print(f"  🤖 Agent chatted: {response_message.content[:50]}...")
                messages.append({"role": "user", "content": "Focus. Use a tool to progress."})

        total_rewards.append(final_reward)

    print(f"\n{'='*60}\n  SUMMARY: Average Reward: {sum(total_rewards)/len(total_rewards):.2f}\n{'='*60}")

if __name__ == "__main__":
    main()