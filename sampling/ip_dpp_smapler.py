import numpy as np
from dppy.finite_dpps import FiniteDPP
import torch
from numpy.random import RandomState
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from sampling.dpp_set import DPPSet
import torch.nn.functional as F


class IPDPPSampler(nn.Module):

        def __init__(self, data, targets, class_num=10, batch_size=64, transform=None):
            super().__init__()
            self.data = data
            self.targets = targets
            self.class_num = class_num
            self.batch_size = batch_size
            self.transform = transform

        def dataset_separation(self):
            """
            Separate data by label and return a list of subsets.

            Returns:
                list: A list of `DPPSet` instances, one for each class.
            """
            sets = []

            # Iterate over class labels and separate data by label
            for class_label in range(self.class_num):
                # Find indices where targets match the current class label
                indices = np.where(self.targets == class_label)

                # Extract subset of data and labels
                sub_data = self.data[indices]
                sub_labels = self.targets[indices]

                # Create a DPPSet for the subset
                sub_set = DPPSet(sub_data, sub_labels, self.transform)
                sets.append(sub_set)

            return sets

        def compute_probabilities(self, model, dataloader, device="cuda"):
            """
            Compute p(i), the probability of correctly predicting the ground truth label for each image.

            Args:
                model (torch.nn.Module): The trained classifier (e.g., ResNet-18).
                dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
                device (str): Device to perform computation ("cuda" or "cpu").

            Returns:
                torch.Tensor: Probabilities p(i) for all images in the dataset.
            """
            model.eval()
            probabilities = []

            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images)

                    # Softmax to get probabilities
                    probs = F.softmax(outputs, dim=1)

                    # Collect probabilities for the ground truth labels
                    p_i = probs[torch.arange(probs.size(0)), labels]
                    probabilities.append(p_i)

            # Concatenate probabilities for all batches
            return torch.cat(probabilities, dim=0)

        def construct_symmetric_matrix(self, P, device="cuda"):
            """
            Constructs the symmetric stochastic matrix S based on the given probabilities.

            Args:
                P (torch.Tensor): A tensor of shape (N,) containing p(i), the probability of correctly predicting each element.
                device (str): Device to perform the computation (e.g., "cuda" or "cpu").

            Returns:
                torch.Tensor: The symmetric stochastic matrix S of shape (N, N).
            """

            # Move probabilities to the specified device
            P = P.to(device)

            # Step 1: Compute outer product to get p(i) * p(j)
            S = P.unsqueeze(0) * P.unsqueeze(1)  # Shape: (N, N)

            # Normalize off-diagonal elements
            N = S.size(0)
            S = S / (N ** 2)

            # Step 2: Compute off-diagonal sums for all rows
            row_sums = S.sum(dim=1)  # Sum of each row (including diagonal)
            diag_values = 1 - (row_sums - S.diagonal())  # Compute diagonal values

            # Step 3: Update the diagonal elements using advanced indexing
            diagonal_indices = torch.arange(N, device=device)
            S[diagonal_indices, diagonal_indices] = diag_values

            return S

        def forward(self, model, num_per_classes = [100 for i in range(10)]):
            rng = RandomState(42)

            sets = self.dataset_separation()
            print("Start IP-DPP sampling ...")
            for i, subset in enumerate(reversed(sets)):

                print("IP-DPP sampling: {}/{}".format(i + 1, self.class_num))

                k = num_per_classes[i]
                if len(subset.labels) <= k:
                    continue

                data_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

                P = self.compute_probabilities(model, data_loader)
                S = self.construct_symmetric_matrix(P)

                DPP = FiniteDPP('likelihood', **{'L': S.detach().cpu().numpy()})

                DPP.sample_mcmc_k_dpp(size=k, random_state=rng)

                indices = DPP.list_of_samples[0][0]

                subset.images = subset.images[indices]
                subset.labels = subset.labels[indices]

            print("Complete IP-DPP sampling!")

            dataset = ConcatDataset(sets)
            return dataset