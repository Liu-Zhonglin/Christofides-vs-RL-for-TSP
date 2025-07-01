import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# We can reuse our Actor model architecture
from actor_critic_model import Actor
from christofides import solve_tsp_christofides

# --- Hyperparameters for the 100-City Problem ---
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_CITIES = 100
BATCH_SIZE = 128
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-4


def calculate_tour_length_torch(cities, tour_indices):
    """Calculates the total length of a batch of tours using PyTorch."""
    batch_size, num_cities, _ = cities.shape
    tour_cities = torch.gather(cities, 1, tour_indices.unsqueeze(-1).expand(batch_size, num_cities, 2))
    tour_distances = (tour_cities - torch.roll(tour_cities, 1, dims=1)).norm(p=2, dim=2)
    return tour_distances.sum(dim=1)


def train_hybrid_reinforce_100():
    """
    Trains the Hybrid REINFORCE model on 100-city problems.
    """
    print(f"--- Starting Hybrid REINFORCE Model Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Actor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    avg_tour_lengths = []
    best_avg_length = float('inf')
    best_epoch = 0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training Hybrid REINFORCE (100 Cities)"):
        # 1. Generate a batch of numpy problems for Christofides
        cities_np_batch = [np.random.rand(NUM_CITIES, 2) for _ in range(BATCH_SIZE)]

        # 2. For each problem, get the tour proposed by Christofides
        christofides_tours = [solve_tsp_christofides(c)[0] for c in cities_np_batch]

        # 3. Create the input batch for the RL model by reordering the cities
        ordered_cities_list = [cities_np_batch[i][tour[:-1]] for i, tour in enumerate(christofides_tours)]
        cities_torch = torch.from_numpy(np.array(ordered_cities_list)).float().to(device)

        # --- Standard REINFORCE Training Loop ---
        model.train()
        tour_indices, tour_log_probs, _ = model(cities_torch)

        reward = calculate_tour_length_torch(cities_torch, tour_indices)

        loss = (reward * tour_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_length = reward.mean().item()
        avg_tour_lengths.append(avg_length)

        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_epoch = epoch
            # Save the new champion model for 100 cities
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving model... **")
            torch.save(model.state_dict(), "best_hybrid_reinforce_model_100.pth")

    print(f"\n--- Hybrid REINFORCE (100 Cities) Training Finished ---")
    print(f"The best model was saved from epoch {best_epoch + 1} with an average length of {best_avg_length:.4f}")

    # Plot and save the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Hybrid REINFORCE Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.legend()
    plt.savefig("hybrid_reinforce_learning_curve_100.png")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(400)  # Use a new seed for this experiment
    train_hybrid_reinforce_100()
