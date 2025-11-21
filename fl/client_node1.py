import os
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from .model import HealthRiskNet
from .data_utils import load_node_1_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Node1Client(fl.client.NumPyClient):
    def __init__(self):
        X_train, y_train, X_val, y_val = load_node_1_data(BASE_DIR)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        input_dim = X_train.shape[1]
        self.model = HealthRiskNet(input_dim=input_dim)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config: Dict[str, str]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 64))

        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            print(f"[Node1] Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}")

        return self.get_parameters({}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            preds = self.model(self.X_val)
            loss = self.loss_fn(preds, self.y_val).item()
            acc = ((preds > 0.5) == self.y_val).float().mean().item()

        print(f"[Node1] Eval loss={loss:.4f}, acc={acc:.4f}")
        return loss, len(self.X_val), {"accuracy": acc}


def main():
    client = Node1Client()
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8085",
        client=client,
    )


if __name__ == "__main__":
    main()
