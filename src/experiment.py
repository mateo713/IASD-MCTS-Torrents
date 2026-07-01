"""Legacy compatibility exports for experiment entrypoints."""

from experiments.human_play import run_human_vs_bot
from experiments.self_play import run_offline_action_sampling_smoke

__all__ = ["run_human_vs_bot", "run_offline_action_sampling_smoke"]