import os
from typing import Callable

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays

from .model import HealthRiskNet

# -----------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, "global_model.pth")


# -----------------------------------------------------------
# GLOBAL EVALUATION FUNCTION
# -----------------------------------------------------------
def get_evaluate_fn() -> Callable:
    """
    Global evaluation on dummy validation data.
    Replace later with real central validation set.
    """
    X_val = torch.randn(256, 8)
    y_val = (torch.rand(256, 1) > 0.5).float()

    def evaluate(server_round: int, parameters, config):
        model = HealthRiskNet(input_dim=8)

        # Convert Flower parameters → PyTorch state_dict
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            preds = model(X_val)
            loss = nn.BCELoss()(preds, y_val).item()
            acc = ((preds > 0.5) == y_val).float().mean().item()

        print(f"[Server] ROUND {server_round} — Global Eval loss={loss:.4f}, acc={acc:.4f}")
        return loss, {"accuracy": acc}

    return evaluate


# -----------------------------------------------------------
# CUSTOM STRATEGY WITH MODEL SAVING
# -----------------------------------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Extends FedAvg:
    - Aggregates normally
    - Saves global model after each round
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:
            # Convert Flower "Parameters" → list of ndarrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Load into PyTorch model
            model = HealthRiskNet(input_dim=8)
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)

            # Save global FL model
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"\n[SAVED] Global model after round {server_round} → {GLOBAL_MODEL_PATH}\n")

        return aggregated_parameters, metrics


# -----------------------------------------------------------
# SERVER MAIN
# -----------------------------------------------------------
def main():
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8085",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )


if __name__ == "__main__":
    main()
