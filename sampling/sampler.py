import collections
import random
from torch.utils.data import Subset


def sample_balanced_subset(dataset, samples_per_class=50):
    class_indices = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    balanced_indices = []

    for class_id, indices in class_indices.items():
        if len(indices) < samples_per_class:
            balanced_indices.extend(indices)
        else:
            balanced_indices.extend(random.sample(indices, samples_per_class))

    balanced_subset = Subset(dataset, balanced_indices)
    return balanced_subset