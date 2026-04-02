import sys
sys.path.append('.')
from server.support_env_environment import SupportEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

env = SupportEnvironment()
env.reset()
print("Initialized")

try:
    obs = env.step(CallToolAction(tool_name="read_ticket", arguments={"thought": "12345 67890 12345"}))
    print("Step returned:", obs)
except Exception as e:
    import traceback
    traceback.print_exc()
