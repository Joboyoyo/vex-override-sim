"""AI agents for VEX Override.

Currently:
- ScriptedBot: simple state-machine match-load runner.

Future:
- Gymnasium env wrapper for RL
- CleanRL PPO baselines
"""

from .scripted_bot import ScriptedBot

__all__ = ["ScriptedBot"]
