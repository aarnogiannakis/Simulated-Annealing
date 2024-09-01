# Simulated-Annealing for CloudComp Scheduling

This project is my submission for the course 42137 Optimization using Metaheuristics, taught by Thomas Jacob Riis Stidsen and Dario Pacino at DTU.

This script implements a Simulated Annealing algorithm with Reheating for solving the CloudComp Scheduling problem. The goal is to minimize the total makespan of the batch, optimizing the assignment and scheduling of operations across processors.


### Key aspects of this algorithm include:
- **Neighborhood Operator:** The algorithm utilizes swap operations between jobs to generate neighboring solutions, aiming to explore various job orderings within processors.
- **Termination Criteria:** The algorithm is designed to run within a predefined time limit, ensuring that the solution process completes within an acceptable timeframe.
- **Step Criterion**: The selection of the next move is based on a probabilistic approach. If a move improves the current solution, it is always accepted. Otherwise, it may still be accepted based on a probability that decreases with temperature.
- **Cost Function:** The algorithm operates on a duration matrix and a processor assignment matrix, both derived from the input file. The cost of a solution is determined by the makespan, which is the maximum completion time across all processors.
- **Initialization:** The initial solution is generated by assigning operations based on randomly generated permutations of job orders, providing a diverse starting point for the optimization.
- **Simulated Annealing:** The algorithm dynamically adjusts the temperature according to a cooling rate (α), balancing the exploration of new solutions and the exploitation of known good solutions over the course of iterations.
- **Reheating:** If no improvement is observed after a predefined number of iterations, the algorithm reheats by resetting the temperature to its initial value, allowing for renewed exploration of the solution space.
