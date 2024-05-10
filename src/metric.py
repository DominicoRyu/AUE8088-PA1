from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)

        for cls in range(self.num_classes):
            tp = torch.sum((preds == cls) & (target == cls))
            fp = torch.sum((preds == cls) & (target != cls))
            fn = torch.sum((preds != cls) & (target == cls))
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Predictions and targets should have the same shape"

        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
