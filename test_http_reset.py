"""Test script to verify HTTP reset cycles through tasks correctly by calling read_ticket."""
import httpx
import time

BASE = "http://localhost:8000"

print("Testing HTTP /reset iteration using read_ticket...")

for i in range(1, 6):
    try:
        # Reset to advance task
        httpx.post(f"{BASE}/reset", timeout=5).raise_for_status()
        
        # Step: Read ticket to see what task we are on
        # ADDED: "thought" parameter to satisfy the updated FastMCP tool schema
        payload = {
            "action": {
                "type": "call_tool", 
                "tool_name": "read_ticket", 
                "arguments": {
                    "thought": "Reading ticket to verify task cycle."
                }
            }
        }
        
        r = httpx.post(f"{BASE}/step", json=payload, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        obs = data.get("observation", {})
        result = obs.get("result", {})
        
        if isinstance(result, dict) and "content" in result:
            text = result["content"][0]["text"]
        elif hasattr(result, "data"):
            text = result.data
        else:
            text = str(result)
            
        print(f"Call {i} Ticket Preview: {text[:60]}...")
        time.sleep(0.5)
    except Exception as e:
        print(f"Call {i} failed: {e}")