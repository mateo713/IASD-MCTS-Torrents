import asyncio
import logging
from config import AppConfig
from experiments.showdown import run_showdown_match
from strategies.expectiminimax import ExpectiminimaxStrategy
from strategies.mcts import MctsStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

async def main():
    config = AppConfig(
        websocket_uri='ws://localhost:8000/showdown/websocket',
        battle_format='gen5randombattle',
        username='itest_me',
        password=None,
        rng_seed=0,
        strategy_name='m',
    )
    outcome = await asyncio.wait_for(
        run_showdown_match(MctsStrategy, ExpectiminimaxStrategy, config=config, session_tag='me-test'),
        timeout=180,
    )
    print('OUTCOME', outcome)
    print('AS_DICT', outcome.as_dict())

asyncio.run(main())
