In order to run the code:

Install Node.JS and clone the pokemon-showdown source code: https://github.com/smogon/pokemon-showdown

You can then start the Showdown local server by going in the showdown directory and running 
node pokemon-showdown start
it should display a few lines of code (normally 4) and say it's available on localhost:8000

You will also need to install poke-engine, with
pip install poke-engine --config-settings="build-args=--features poke-engine/gen5 --no-default-features"

Then, you can run the battler's code. 
The file is src/main.py

It has 3 modes: play against a human, make two strategies play a battle, or run a tournament. 
The strategies that can be specified are
random, first legal move, max-damage, heuristic, expectiminimax, mcts
the associated command line shorthands are
r, f, d, h, e, m

human mode: one can easily create a showdown account to then challenge mcts_bot in gen5randombattles, the command is
python main.py -m=human -s={insert strategy shorthand here}

match mode: choose two strategies, and make them battle, the result will be printed out eventually.
python main.py -m=match -a={first strategy shorthand} -b={second strategy shorthand}

tournament mode: create a tournament with some of the strategies and a certain number of matches
python main.py -m=tournament -l={comma separated list of strategy shorthands, defining the participants} -n={nuumber of matches played for each pair of strategies}
example: python main.py -m=tournament -l=h,e,m -n=2
There are other arguments, mainly to limit the number of concurrent matches as that can make things more unstable, so it is not recommended to change them, though the list of arguments can be obtained with python main.py -help.
