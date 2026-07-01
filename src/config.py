from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_WEBSOCKET_URI = "ws://127.0.0.1:8000/showdown/websocket"
DEFAULT_BATTLE_FORMAT = "gen5randombattle"
DEFAULT_USERNAME = "mcts_bot"


@dataclass(slots=True)
class AppConfig:
    websocket_uri: str = DEFAULT_WEBSOCKET_URI
    battle_format: str = DEFAULT_BATTLE_FORMAT
    username: str = DEFAULT_USERNAME
    password: str | None = None
    rng_seed: int = 0
    strategy_name: str = "r"
    team: str | None = None
    team_one: str | None = None
    team_two: str | None = None
    show_constraints: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        seed_text = os.getenv("PB_RNG_SEED", "0")
        try:
            seed_value = int(seed_text)
        except ValueError:
            seed_value = 0

        return cls(
            websocket_uri=os.getenv("PB_WEBSOCKET_URI", DEFAULT_WEBSOCKET_URI),
            battle_format=os.getenv("PB_BATTLE_FORMAT", DEFAULT_BATTLE_FORMAT),
            username=os.getenv("PB_USERNAME", DEFAULT_USERNAME),
            password=os.getenv("PB_PASSWORD") or None,
            rng_seed=seed_value,
            strategy_name=os.getenv("PB_STRATEGY", "r"),
            team=os.getenv("PB_TEAM") or None,
            team_one=os.getenv("PB_TEAM_ONE") or None,
            team_two=os.getenv("PB_TEAM_TWO") or None,
            show_constraints=(os.getenv("PB_SHOW_CONSTRAINTS", "0") in {"1", "true", "True"}),
        )
