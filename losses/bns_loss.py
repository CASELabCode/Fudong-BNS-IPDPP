import torch
import torch.nn as nn
import torch.nn.functional as F

class BNSLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        # Negative mask for all pairs
        self.register_buffer(
            "negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        )

    def forward(self, emb_i, emb_j, labels_i, labels_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are positive pairs.
        labels_i and labels_j are their corresponding labels.
        """
        # Normalize embeddings
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate embeddings
        representations = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        # Concatenate labels and compute pairwise label equality
        labels = torch.cat([labels_i, labels_j], dim=0)
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Positive mask (pairs with the same label but not self-pairs)
        positives_mask = label_matrix * (~torch.eye(self.batch_size * 2, dtype=bool, device=labels.device)).float()
        positives = (positives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1) / positives_mask.sum(dim=1).clamp(min=1)

        # Negative mask (pairs with different labels)
        negatives_mask = self.negatives_mask * (~label_matrix).float()
        negatives = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        # Loss computation
        true_loss = F.logsigmoid(positives)
        noise_loss = F.logsigmoid(-negatives.mean(dim=1))

        loss = -(true_loss + noise_loss).mean()

        return loss