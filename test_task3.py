import sys
import os
import httpx
# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))
from client import SupportEnv
from openenv.core.env_server.mcp_types import CallToolAction

ENV_URL = "http://localhost:8000"

def step(env, tool: str, **kwargs):
    return env.step(CallToolAction(tool_name=tool, arguments=kwargs))

def run_task_3():
    with SupportEnv(base_url=ENV_URL).sync() as env:
        print("Resetting for Task 3...")
        r = env.reset(options={"task_idx": 2})
        print(f"Reset done. Done={r.done}, Reward={r.reward}")
        
        print("Step 1: read_ticket")
        ticket_obs = step(env, "read_ticket", thought="Read ticket for request identification.")
        ticket_text = ticket_obs.observation.result.data if hasattr(ticket_obs.observation.result, 'data') else str(ticket_obs.observation.result)
        print(f"Ticket: {ticket_text}")
        
        print("Step 2: check_billing")
        # Extract UID from ticket
        import re
        m = re.search(r"USR\d{3}", ticket_text)
        uid = m.group(0) if m else "USR001"
        print(f"Using UID: {uid}")
        
        db_obs = step(env, "check_billing", thought="Check billing status for user.", user_id=uid)
        db_text = db_obs.observation.result.data if hasattr(db_obs.observation.result, 'data') else str(db_obs.observation.result)
        print(f"DB Obs: {db_text}")
        
        print("Step 3: search_knowledge_base")
        kb_obs = step(env, "search_knowledge_base", thought="Search KB for policy.", query="suspended account")
        kb_text = kb_obs.observation.result.data if hasattr(kb_obs.observation.result, 'data') else str(kb_obs.observation.result)
        print(f"KB Obs: {kb_text}")
        
        print("Step 4: escalate_ticket")
        sr = step(env, "escalate_ticket", thought="After verifying the KB rules and account state, I can now escalate per protocol.", department="security")
        print(f"Final Reward: {sr.reward:.2f}")
        print(f"Done: {sr.done}")
        if not sr.done:
            print(f"EPISODE NOT DONE! Observation: {sr.observation.result}")

if __name__ == "__main__":
    run_task_3()
