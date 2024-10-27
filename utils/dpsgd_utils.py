from opacus import PrivacyEngine
import torch
import copy

def find_noise_multiplier(model, target_epsilon, target_delta, train_loader, max_grad_norm, num_epochs):
    """
    Find a suitable noise multiplier that satisfies the target (ε, δ) privacy budget.

    Args:
        model: The neural network model (e.g., net_glob).
        target_epsilon (float): The target epsilon for differential privacy.
        target_delta (float): The target delta for differential privacy.
        train_loader: The DataLoader for the training dataset.
        max_grad_norm (float): The maximum gradient norm for clipping.
        num_epochs (int): The number of training epochs.

    Returns:
        float: The noise multiplier that satisfies the target privacy budget.
    """
    # Start with an initial noise multiplier
    noise_multiplier = 1.0
    step_size = 0.1  # Increment step for noise multiplier

    # Create a copy of the model to avoid altering the original
    model_copy = copy.deepcopy(model)
    model_copy.train()

    # Optimizer setup
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.01)

    # Searching for a suitable noise multiplier
    while True:
        # Initialize PrivacyEngine
        privacy_engine = PrivacyEngine()

        # Make the model private
        model_copy, optimizer, train_loader_private = privacy_engine.make_private(
            module=model_copy,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        # Compute privacy spent after num_epochs
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=target_delta)
        print(f'Privacy budget after training: ε = {epsilon:.2f}, δ = {target_delta} for α = {best_alpha}')

        # If the computed epsilon meets the target, break out of the loop
        if epsilon <= target_epsilon:
            break

        # If not, increase the noise multiplier and try again
        noise_multiplier += step_size

        # Detach the privacy engine before the next iteration
        privacy_engine.detach()

    return noise_multiplier
