import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from actor_critic_model import Actor, Critic
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


def train_actor_critic_100():
    """
    Trains the Actor-Critic model on 100-city problems, using Christofides' solution
    to structure the input sequence and provide a reward baseline.
    """
    print(f"--- Starting Actor-Critic Hybrid Model Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    actor = Actor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    critic = Critic(hidden_dim=HIDDEN_DIM).to(device)

    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)

    avg_tour_lengths = []
    best_avg_length = float('inf')
    best_epoch = 0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training Actor-Critic (100 Cities)"):
        # 1. Generate a batch of numpy problems for Christofides
        cities_np_batch = [np.random.rand(NUM_CITIES, 2) for _ in range(BATCH_SIZE)]

        # 2. Get Christofides tours to create ordered input and baseline rewards
        christofides_tours = [solve_tsp_christofides(c) for c in cities_np_batch]
        ordered_cities_list = [cities_np_batch[i][tour[0][:-1]] for i, tour in enumerate(christofides_tours)]
        cities_torch = torch.from_numpy(np.array(ordered_cities_list)).float().to(device)

        # Get baseline lengths from Christofides
        lengths_christofides = torch.tensor([tour[1] for tour in christofides_tours], dtype=torch.float32,
                                            device=device)

        # --- Actor-Critic Training Loop ---
        actor.train()
        critic.train()

        tour_indices, tour_log_probs, final_hidden_state = actor(cities_torch)

        value = critic(final_hidden_state).squeeze()
        reward = calculate_tour_length_torch(cities_torch, tour_indices)

        advantage = lengths_christofides.detach() - reward  # Reward is how much we BEAT the baseline
        actor_loss = (advantage * tour_log_probs).mean()
        critic_loss = nn.functional.mse_loss(value, reward)
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_length = reward.mean().item()
        avg_tour_lengths.append(avg_length)

        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_epoch = epoch
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving models... **")
            torch.save(actor.state_dict(), "best_ac_actor_model_100.pth")
            torch.save(critic.state_dict(), "best_ac_critic_model_100.pth")

    print(f"\n--- Actor-Critic (100 Cities) Training Finished ---")
    print(f"The best model was saved from epoch {best_epoch + 1} with an average length of {best_avg_length:.4f}")

    # Plot and save the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Actor-Critic Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.legend()
    plt.savefig("ac_learning_curve_100.png")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(500)  # Use a new seed for this final experiment
    train_actor_critic_100()
