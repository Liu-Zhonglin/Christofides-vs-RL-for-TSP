# A Comparative Study of Classical Heuristics and Reinforcement Learning for the TSP

[**View the Full Project Report (PDF)**](https://liu-zhonglin.github.io/Christofides-vs-RL-for-TSP/Project%20Report/Report.pdf)

**Course:** HKU MATH3999: Directed Studies in Mathematics  
**Author:** Liu Zhonglin  
**Supervisor:** Prof. Zang Wenan

---

## Project Overview

This repository archives a directed study project that implements and compares multiple methodologies for solving the Traveling Salesman Problem (TSP). The core of the study is a robust, quantitative evaluation of four distinct approaches on 30-city TSP instances:

1.  **Classical Heuristic:** A scalable implementation of the Christofides algorithm.
2.  **Pure Reinforcement Learning:** A Pointer Network trained from scratch using the REINFORCE algorithm.
3.  **Hybrid RL (REINFORCE):** A Pointer Network trained on input sequences that were pre-ordered using the Christofides algorithm.
4.  **Hybrid RL (Actor-Critic):** An advanced version of the hybrid model trained with a more stable Actor-Critic algorithm.

The goal was to quantify the performance trade-offs between classical, theory-driven algorithms and modern, learning-based approaches, and to investigate whether hybridizing these methods could yield superior results.

## Final Results

The definitive results were obtained by evaluating each method over **1000 unique, randomly generated 30-city TSP problems**. The final average performance is summarized below.

| Metric                 | Christofides | Pure RL         | Hybrid (REINFORCE) | Hybrid (Actor-Critic) |
| ---------------------- | ------------ | --------------- | ------------------ | --------------------- |
| **Avg. Tour Length** | **5.3280** | 11.9893         | 10.4065            | **10.3216** |
| **Avg. Comp. Time (s)**| **0.0005** | 0.0232          | 0.0228             | 0.0229                |

## Conclusion

The study yielded two primary conclusions:

1.  **Classical algorithms remain superior** for this well-defined problem, with the Christofides algorithm decisively outperforming all RL models in both solution quality and speed.
2.  **Hybridization is a powerful strategy.** Both hybrid RL models, which leveraged the structural knowledge from Christofides to order their input, found significantly better solutions than the pure RL agent. The more advanced Actor-Critic algorithm demonstrated a further, measurable improvement over the simpler REINFORCE hybrid.

This work suggests that the most promising direction for applying RL to complex optimization problems lies in creating sophisticated hybrid systems where classical heuristics guide the powerful, flexible learning capabilities of neural networks.

## License

This project is licensed under the MIT License.