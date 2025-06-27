import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pointer_network import PointerNetwork
from christofides import solve_tsp_christofides

# --- Hyperparameters for the 30-City Problem ---
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_CITIES = 30
BATCH_SIZE = 128
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-4


def calculate_tour_length_torch(cities, tour_indices):
    """Calculates the total length of a batch of tours using PyTorch."""
    batch_size, num_cities, _ = cities.shape
    tour_cities = torch.gather(cities, 1, tour_indices.unsqueeze(-1).expand(batch_size, num_cities, 2))
    tour_distances = (tour_cities - torch.roll(tour_cities, 1, dims=1)).norm(p=2, dim=2)
    return tour_distances.sum(dim=1)


def train_hybrid_30():
    """
    Trains the RL model on 30-city problems, using Christofides' solution
    to structure the input sequence for the Pointer Network.
    Includes logic to save the best-performing model.
    """
    print(f"--- Starting Hybrid Model Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PointerNetwork(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    avg_tour_lengths = []
    # --- NEW: Keep track of the best score ---
    best_avg_length = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # --- NEW HYBRID DATA GENERATION ---
        cities_np_batch = [np.random.rand(NUM_CITIES, 2) for _ in range(BATCH_SIZE)]
        christofides_tours = [solve_tsp_christofides(c)[0] for c in cities_np_batch]
        ordered_cities_list = []
        for i in range(BATCH_SIZE):
            tour = christofides_tours[i][:-1]
            ordered_cities_list.append(cities_np_batch[i][tour])

        cities_torch = torch.from_numpy(np.array(ordered_cities_list)).float().to(device)

        # --- Standard Training Loop ---
        model.train()
        tour_indices_rl, tour_log_probs = model(cities_torch)

        lengths_rl = calculate_tour_length_torch(cities_torch, tour_indices_rl)

        loss = (lengths_rl * tour_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_length = lengths_rl.mean().item()
        avg_tour_lengths.append(avg_length)

        # --- NEW: Save-on-best logic ---
        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_epoch = epoch
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving model... **")
            torch.save(model.state_dict(), "best_hybrid_reinforce_model_30.pth")

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Avg Tour Length: {avg_length:.4f}")

    print("--- 30-City Hybrid Training Finished ---")
    print(f"The best model was saved from epoch {best_epoch + 1} with an average length of {best_avg_length:.4f}")

    # The saved model name is updated to reflect it's the "best" one.
    model_save_path = "best_hybrid_reinforce_model_30.pth"
    # Note: the model is already saved in the loop, this is just for confirmation.
    print(f"Best model saved to {model_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Hybrid RL Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.legend()
    plt.savefig("hybrid_reinforce_learning_curve_30.png")
    plt.show()


if __name__ == '__main__':
    # Using the same seed as the first hybrid run for a fair comparison
    torch.manual_seed(42)
    train_hybrid_30()