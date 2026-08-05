# =============================================================================
# Reference PPO agent package (AGENT=ppo)
# =============================================================================
# Everything specific to the built-in PPO submission lives here:
#   agent.py      - the ContinualAgent adapter (entry point of this package)
#   trainer.py    - the jitted single-task PPO loop
#   networks.py   - CNN / MLP torsos + Actor/Critic heads + AgentParams
#   eval.py       - checkpoint-based eval + the mid-training curve eval
#   continual/    - CL methods (ft/ewc/agem/packnet) behind the CLMethod interface

from agents.ppo.agent import PPOCRLAgent, PPOState

__all__ = ["PPOCRLAgent", "PPOState"]
