# A Comparative Study of Classical Heuristics and Reinforcement Learning for the TSP

[**View the Full Project Report (PDF)**](https://liu-zhonglin.github.io/Christofides-vs-RL-for-TSP/Project%20Report/Report.pdf)

**Course:** HKU MATH3999: Directed Studies in Mathematics  
**Author:** Liu Zhonglin  
**Supervisor:** Prof. Zang Wenan

---

## Project Overview

This repository archives a directed study project that implements and compares multiple methodologies for solving the Traveling Salesman Problem (TSP). The core of the study is a robust, quantitative evaluation of five distinct approaches on both 30-city and 100-city TSP instances:

1. **Classical Heuristic (Christofides):** A scalable implementation of the Christofides algorithm.
2. **Classical Heuristic (Nearest Insertion):** A greedy constructive heuristic.
3. **Pure Reinforcement Learning:** A Pointer Network trained from scratch using the REINFORCE algorithm.
4. **Hybrid RL (REINFORCE):** A Pointer Network trained on input sequences that were pre-ordered using the Christofides algorithm.
5. **Hybrid RL (Actor-Critic):** An advanced version of the hybrid model trained with a more stable Actor-Critic algorithm.

The goal was to quantify the performance trade-offs between classical, theory-driven algorithms and modern, learning-based approaches, and to investigate whether hybridizing these methods could yield superior results, especially at a larger scale.

## Final Results

The definitive results were obtained by evaluating each method over **1000 unique, randomly generated problems** for both the 30-city and 100-city scales. The final average performance is summarized below.

#### Average Tour Length Comparison

| Method                  | 30 Cities (Avg. Length) | 100 Cities (Avg. Length) |
| ----------------------- | ----------------------- | ------------------------ |
| **Nearest Insertion** | **5.2107** | **9.3125** |
| **Christofides** | 5.3300                  | 9.3701                   |
| **Hybrid (Actor-Critic)** | 10.3699                 | **9.3705** |
| **Hybrid (REINFORCE)** | 10.5369                 | 26.9063                  |
| **Pure RL** | 11.9242                 | 32.4112                  |

## Key Conclusions

This study yielded three primary conclusions:

1. **Classical Heuristics Remain Superior:** For the standard Euclidean TSP, classical algorithms like Nearest Insertion and Christofides are dominant, providing the best solutions with unparalleled speed.

2. **Hybridization is a Powerful Strategy:** The performance of RL agents was dramatically improved when they were provided with a high-quality input sequence structured by the Christofides algorithm. This confirms that integrating classical domain knowledge is a highly effective strategy.

3. **Advanced Training is Critical for Scaling:** The most striking result was the performance of the Actor-Critic hybrid model on the 100-city problem. It achieved a tour length statistically identical to the classical heuristics, a feat the simpler REINFORCE hybrid could not come close to. This demonstrates that stable, advanced training algorithms are essential for RL models to learn competitive policies for complex, large-scale problems.

## License

This project is licensed under the MIT License.