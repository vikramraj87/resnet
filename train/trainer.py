from torch import no_grad, Tensor
import torch


class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(x)
        loss = self.criterion(logits, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @no_grad()
    def validation_step(self, x, y):
        logits = self.model(x)
        loss = self.criterion(logits, y)

        accuracy, n_correct = self._accuracy(logits, y)
        return loss.item(), accuracy

    @no_grad()
    def test_step(self, x, y):
        self.model.eval()
        logits = self.model(x)
        _, n_correct = self._accuracy(logits, y)
        return n_correct

    @staticmethod
    def _accuracy(logits: Tensor, y: Tensor):
        assert logits.shape[0] == y.shape[0]

        n = logits.shape[0]
        y_pred = torch.argmax(logits, 1)
        n_correct = (y == y_pred).sum().item()

        accuracy = float(n_correct) / float(n)
        return accuracy, n_correct
