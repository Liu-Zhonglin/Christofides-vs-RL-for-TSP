# A Comparative Study of Classical Heuristics and Reinforcement Learning for the TSP

**Course:** HKU MATH3999: Directed Studies in Mathematics  
**Author:** Liu Zhonglin  
**Supervisor:** Prof. Zang Wenan

---

## Overview

This repository contains the full implementation for a directed study project comparing classical algorithms and deep reinforcement learning for solving the Traveling Salesman Problem (TSP). The project evaluates four distinct methods on 30-city TSP instances:

1.  A **Classical Heuristic** based on the Christofides algorithm (using a scalable greedy matching approach).
2.  A **Pure Reinforcement Learning** model (Pointer Network) trained from scratch with the REINFORCE algorithm.
3.  A **Hybrid REINFORCE Model** that uses the Christofides tour to structure the input sequence for the RL agent.
4.  An **Advanced Hybrid Actor-Critic Model** that uses the same structured input but is trained with a more stable A2C-style algorithm.

The primary finding is that while the classical Christofides algorithm remains superior in both solution quality and speed, the method of integrating its structural knowledge into the RL training process significantly enhances performance over a pure learning approach.

## Final Results

The following table presents the final, robust average results from evaluating each method over **1000 unique, randomly generated 30-city TSP problems**.

| Metric                 | Christofides | Pure RL         | Hybrid (REINFORCE) | Hybrid (Actor-Critic) |
| ---------------------- | ------------ | --------------- | ------------------ | --------------------- |
| **Avg. Tour Length** | **5.3280** | 11.9893         | 10.4065            | **10.3216** |
| **Avg. Comp. Time (s)**| **0.0005** | 0.0232          | 0.0228             | 0.0229                |


## File Structure

The project is organized into several key Python scripts:

-   `christofides.py`: A scalable implementation of the classical Christofides algorithm.
-   `actor_critic_model.py`: Contains the PyTorch class definitions for the `Actor` (Pointer Network) and `Critic` models.
-   `train_pure_rl_30.py`: The script to train the "Pure RL" model from scratch.
-   `train_hybrid_30.py`: The script to train the "Hybrid (REINFORCE)" model.
-   `train_actor_critic.py`: The script to train the final, advanced "Hybrid (Actor-Critic)" model.
-   `final_robust_evaluation.py`: The master script to run the final 4-way comparison and generate the results table.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Christofides-vs-RL-for-TSP.git](https://github.com/YourUsername/Christofides-vs-RL-for-TSP.git)
    cd Christofides-vs-RL-for-TSP
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:** A `requirements.txt` file is included for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

You can replicate the entire project by running the training scripts followed by the final evaluation.

1.  **Train the Models (Optional, as pre-trained models can be provided):**
    ```bash
    # Train the Pure RL model
    python train_pure_rl_30.py

    # Train the REINFORCE Hybrid model
    python train_hybrid_30.py

    # Train the Actor-Critic Hybrid model
    python train_actor_critic.py
    ```

2.  **Run the Final Evaluation:** This script requires the `.pth` model files from the training scripts to be present in the directory.
    ```bash
    python final_robust_evaluation.py
    ```
    This will run the full 1000-trial evaluation and print the final results table to the console.

## Proposed Future Work

This study revealed several promising avenues for future research:
* **Advanced Architectures:** Replacing the LSTM-based Pointer Network with a Graph Attention Network (GAT) to better capture the geometric structure of the TSP.
* **Hyperparameter Tuning:** Performing a systematic search for optimal hyperparameters (learning rate, hidden dimensions, etc.) to potentially improve RL model performance further.
* **Alternative Hybridization:** Exploring different hybrid methods, such as using the RL agent to select between classical heuristics (e.g., 2-Opt, 3-Opt) to apply to a tour.

## License

This project is licensed under the MIT License.