import torch
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from actor_critic_model import Actor
from christofides import solve_tsp_christofides
from nearest_insertion import solve_tsp_nearest_insertion  # <-- NEW IMPORT


def evaluate_all_methods(num_trials=1000):
    """
    Runs a multi-trial evaluation to compare the average performance of all five models.
    """
    print(f"--- Starting Robust 5-Way Evaluation over {num_trials} Trials (30 Cities) ---")

    # --- Setup ---
    NUM_CITIES = 30
    PURE_RL_PATH = "pure_rl_model_30.pth"
    HYBRID_REINFORCE_PATH = "best_hybrid_reinforce_model_30.pth"
    HYBRID_AC_PATH = "best_ac_actor_model_30.pth"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load all RL models once ---
    pure_model = Actor(input_dim=2, hidden_dim=128).to(device)
    pure_model.load_state_dict(torch.load(PURE_RL_PATH))
    pure_model.eval()

    reinf_hybrid_model = Actor(input_dim=2, hidden_dim=128).to(device)
    reinf_hybrid_model.load_state_dict(torch.load(HYBRID_REINFORCE_PATH))
    reinf_hybrid_model.eval()

    ac_hybrid_model = Actor(input_dim=2, hidden_dim=128).to(device)
    ac_hybrid_model.load_state_dict(torch.load(HYBRID_AC_PATH))
    ac_hybrid_model.eval()

    # --- Lists to store results from all trials ---
    results = {
        'Christofides': {'lengths': [], 'times': []},
        'Nearest Insertion': {'lengths': [], 'times': []},  # <-- NEW CONTENDER
        'Pure RL': {'lengths': [], 'times': []},
        'Hybrid (REINFORCE)': {'lengths': [], 'times': []},
        'Hybrid (Actor-Critic)': {'lengths': [], 'times': []}
    }

    # --- Main Evaluation Loop ---
    for _ in tqdm(range(num_trials), desc="Running Evaluation Trials"):
        test_cities_np = np.random.rand(NUM_CITIES, 2)

        # 1. Christofides
        start_time = time.time()
        c_tour, c_len, _ = solve_tsp_christofides(test_cities_np)
        results['Christofides']['times'].append(time.time() - start_time)
        results['Christofides']['lengths'].append(c_len)

        # 2. Nearest Insertion <-- NEW
        start_time = time.time()
        ni_tour, ni_len = solve_tsp_nearest_insertion(test_cities_np)
        results['Nearest Insertion']['times'].append(time.time() - start_time)
        results['Nearest Insertion']['lengths'].append(ni_len)

        # --- RL Models ---
        test_cities_torch = torch.from_numpy(test_cities_np).float().unsqueeze(0).to(device)
        dist_matrix = squareform(pdist(test_cities_np, 'euclidean'))

        # 3. Pure RL
        start_time = time.time()
        with torch.no_grad():
            tour_indices, _, _ = pure_model(test_cities_torch)
        results['Pure RL']['times'].append(time.time() - start_time)
        tour = tour_indices.squeeze(0).cpu().numpy().tolist() + [0];
        tour[-1] = tour[0]
        results['Pure RL']['lengths'].append(sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

        # Prepare input for Hybrid Models
        initial_ordered_indices = np.array(c_tour[:-1])
        ordered_cities_torch = torch.from_numpy(test_cities_np[initial_ordered_indices]).float().unsqueeze(0).to(device)

        # 4. REINFORCE Hybrid
        start_time = time.time()
        with torch.no_grad():
            rl_perm, _, _ = reinf_hybrid_model(ordered_cities_torch)
        results['Hybrid (REINFORCE)']['times'].append(time.time() - start_time)
        tour = initial_ordered_indices[rl_perm.squeeze(0).cpu().numpy()].tolist() + [0];
        tour[-1] = tour[0]
        results['Hybrid (REINFORCE)']['lengths'].append(
            sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

        # 5. Actor-Critic Hybrid
        start_time = time.time()
        with torch.no_grad():
            rl_perm, _, _ = ac_hybrid_model(ordered_cities_torch)
        results['Hybrid (Actor-Critic)']['times'].append(time.time() - start_time)
        tour = initial_ordered_indices[rl_perm.squeeze(0).cpu().numpy()].tolist() + [0];
        tour[-1] = tour[0]
        results['Hybrid (Actor-Critic)']['lengths'].append(
            sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    # --- Calculate and Report Final Averages ---
    print("\n" + "=" * 145)
    print(f"--- FINAL ROBUST AVERAGE RESULTS ({num_trials} TRIALS, {NUM_CITIES} CITIES) ---")
    print("=" * 145)
    print(
        f"{'Metric':<25} | {'Christofides':<25} | {'Nearest Insertion':<25} | {'Pure RL':<25} | {'Hybrid (REINFORCE)':<25} | {'Hybrid (Actor-Critic)':<25}")
    print("-" * 145)

    avg_lengths = {name: np.mean(data['lengths']) for name, data in results.items()}
    avg_times = {name: np.mean(data['times']) for name, data in results.items()}

    print(
        f"{'Avg. Tour Length':<25} | {avg_lengths['Christofides']:<25.4f} | {avg_lengths['Nearest Insertion']:<25.4f} | {avg_lengths['Pure RL']:<25.4f} | {avg_lengths['Hybrid (REINFORCE)']:<25.4f} | {avg_lengths['Hybrid (Actor-Critic)']:<25.4f}")
    print(
        f"{'Avg. Comp. Time (s)':<25} | {avg_times['Christofides']:<25.4f} | {avg_times['Nearest Insertion']:<25.4f} | {avg_times['Pure RL']:<25.4f} | {avg_times['Hybrid (REINFORCE)']:<25.4f} | {avg_times['Hybrid (Actor-Critic)']:<25.4f}")
    print("-" * 145)


if __name__ == '__main__':
    evaluate_all_methods(num_trials=1000)