# Parameter explanation
We introduce here the main args of the `MCTS*` algorithm.

1. temperature: Search temperature, used to determine the degrees of freedom for generating responses.

2. time_limit: The upper limit of the search time(ms) set in the MCTS framework.

3. iteration_limit: The maximum number of search rounds for exploration.

4. roll_policy: The strategy for Monte Carlo simulation in the MCTS framework, either random or greedy.

5. exploration_constant: Constant for the UCT formula which balances exploration and exploitation.

6. roll_forward_steps: The number of forward steps in the simulation process.

7. end_gate: The lowest value threshold for determining the end of search.

8. branch: The number of branches for node expansion.

9. roll_branch: Number of branches to sample for simulation.

10. inf: The base value of an unvisited node.

11. alpha: Value update weight for Monte Carlo simulation in the MCTS framework.

12. visualize: Whether the results are visualized in a tree diagram.

13. use_case_prompt: Whether to use sample output prompt assisted generation.

14. use_reflection: Whether to use the reflection mechanism.

15. low: The lower bound of the node value.

16. high: The upper bound of the node value.