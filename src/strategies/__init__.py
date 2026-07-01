from strategies.base import Strategy, StrategyFactory
from strategies.baselines import FirstLegalStrategy, RandomUniformStrategy
from strategies.expectiminimax import ExpectiminimaxPlaceholderStrategy
from strategies.expectiminimax import ExpectiminimaxStrategy
from strategies.rules import OneTurnExpectedDamageStrategy
from strategies.rules import HeuristicStrategy
from strategies.mcts import MctsPlaceholderStrategy
from strategies.mcts import MctsStrategy
from strategies.reasoning import ReasoningPlaceholderStrategy

ALL_STRATEGY_FACTORIES: list[StrategyFactory] = [
	RandomUniformStrategy,
	FirstLegalStrategy,
	OneTurnExpectedDamageStrategy,
	HeuristicStrategy,
	ExpectiminimaxStrategy,
	MctsStrategy,
]
STRATEGY_FACTORIES_BY_NAME: dict[str, StrategyFactory] = {
	strategy_factory.name: strategy_factory for strategy_factory in ALL_STRATEGY_FACTORIES
}
STRATEGY_SHORT_NAMES: dict[str, StrategyFactory] = {
	"r": RandomUniformStrategy,
	"f": FirstLegalStrategy,
	"d": OneTurnExpectedDamageStrategy,
	"h": HeuristicStrategy,
	"e": ExpectiminimaxStrategy,
	"m": MctsStrategy,
}
STRATEGY_FACTORIES_BY_SHORT_NAME: dict[str, StrategyFactory] = {
	**STRATEGY_SHORT_NAMES,
	**STRATEGY_FACTORIES_BY_NAME,
}
STRATEGY_SHORT_NAMES_BY_FACTORY: dict[StrategyFactory, str] = {
	factory: short_name for short_name, factory in STRATEGY_SHORT_NAMES.items()
}


def get_strategy_factory(strategy_name: str) -> StrategyFactory:
	try:
		return STRATEGY_FACTORIES_BY_SHORT_NAME[strategy_name]
	except KeyError as exc:
		available = ", ".join(sorted(STRATEGY_FACTORIES_BY_SHORT_NAME))
		raise KeyError(f"Unknown strategy '{strategy_name}'. Available: {available}") from exc


def get_strategy_short_name(strategy_factory: StrategyFactory) -> str:
	return STRATEGY_SHORT_NAMES_BY_FACTORY.get(strategy_factory, strategy_factory.name)


__all__ = [
	"Strategy",
	"StrategyFactory",
	"RandomUniformStrategy",
	"FirstLegalStrategy",
	"OneTurnExpectedDamageStrategy",
	"HeuristicStrategy",
	"ExpectiminimaxStrategy",
	"MctsStrategy",
	"MctsPlaceholderStrategy",
	"ExpectiminimaxPlaceholderStrategy",
	"ReasoningPlaceholderStrategy",
	"ALL_STRATEGY_FACTORIES",
	"STRATEGY_FACTORIES_BY_NAME",
	"STRATEGY_SHORT_NAMES",
	"STRATEGY_FACTORIES_BY_SHORT_NAME",
	"STRATEGY_SHORT_NAMES_BY_FACTORY",
	"get_strategy_factory",
	"get_strategy_short_name",
]
