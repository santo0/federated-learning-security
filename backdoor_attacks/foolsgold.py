import numpy as np
import flwr as fl
import torch

class FoolsgoldServer(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.similarity_matrix = None
        self.trust_scores = None

    def initialize_trust_scores(self, num_clients):
        self.trust_scores = np.ones(num_clients)
        self.similarity_matrix = np.zeros((num_clients, num_clients))

    def compute_similarity(self, updates):
        num_clients = len(updates)
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    self.similarity_matrix[i, j] = np.dot(updates[i], updates[j]) / (np.linalg.norm(updates[i]) * np.linalg.norm(updates[j]))

    def update_trust_scores(self):
        num_clients = len(self.trust_scores)
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    self.trust_scores[i] *= (1 - self.similarity_matrix[i, j])
        self.trust_scores = self.trust_scores / np.max(self.trust_scores)

    def aggregate_fit(self, rnd, results, failures):
        if self.trust_scores is None:
            self.initialize_trust_scores(len(results))
        updates = [res[1] for res in results]
        self.compute_similarity(updates)
        self.update_trust_scores()

        weighted_updates = []
        for i, (client_params, client_size, _) in enumerate(results):
            weight = self.trust_scores[i]
            weighted_updates.append([w * weight for w in client_params])

        averaged_params = [np.mean([weighted_updates[i][j] for i in range(len(weighted_updates))], axis=0) for j in range(len(weighted_updates[0]))]
        return averaged_params, {}
