import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
from config import POS_WEIGHTS_PATH
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

################# CLASS IMBALANCE HANDLING USING BCEWithLogistsLoss ###################

# Compute positive class weights for BCEWithLogitsLoss.
def compute_class_weights(y_train: np.ndarray):

    n_samples, n_labels = y_train.shape
    pos_counts = y_train.sum(axis=0)
    neg_counts = n_samples - pos_counts

    # Avoid division by zero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    np.save(POS_WEIGHTS_PATH, pos_weights)

    return torch.tensor(pos_weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    


############ CLASS IMBALANCE HANDLING USING FOCAL LOSS #####################

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probas = torch.sigmoid(logits)
        
        # Compute focal scaling
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    

class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = FocalLoss(gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss



########### CLASS IMBALANCE HANDLING USING CLASS WEIGHTS #####################

def create_weighted_sampler(labels: np.ndarray):
    # Class counts
    class_counts = labels.sum(axis=0)   # num_classes
    class_freq = class_counts / labels.shape[0]

    # Avoid div-by-zero
    class_weights = 1.0 / (class_freq + 1e-6)

    # Each sample weight = sum of weights of positive labels
    sample_weights = (labels * class_weights).sum(axis=1)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


class ResampledTrainer(Trainer):
    def get_train_dataloader(self):
        """
        Override Trainer to use WeightedRandomSampler for training.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        labels = np.array(self.train_dataset["labels"])
        sampler = create_weighted_sampler(labels)

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )