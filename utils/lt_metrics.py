import torch
from collections import defaultdict


def compute_class_counts(dataset):
    """
    Compute the number of samples per class in the dataset.
    """
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts


def categorize_classes(class_counts, many_threshold=500, few_threshold=200):
    """
    Categorize classes into many-shot, medium-shot, and few-shot based on thresholds.
    """
    many_shot_classes = []
    medium_shot_classes = []
    few_shot_classes = []

    for cls, count in class_counts.items():
        if count > many_threshold:
            many_shot_classes.append(cls)
        elif count < few_threshold:
            few_shot_classes.append(cls)
        else:
            medium_shot_classes.append(cls)

    return many_shot_classes, medium_shot_classes, few_shot_classes


def compute_accuracy(predictions, labels, category_classes, device=torch.device('cuda')):
    """
    Compute accuracy for a specific category of classes.
    """
    mask = torch.isin(labels, torch.tensor(category_classes).to(device))
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0


def evaluate_model(model, dataloader, many_shot_classes, medium_shot_classes, few_shot_classes, device='cuda'):
    """
    Evaluate model and compute many-shot, medium-shot, and few-shot accuracy.
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_predictions.append(preds)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute accuracy for each category
    many_shot_acc = compute_accuracy(all_predictions, all_labels, many_shot_classes)
    medium_shot_acc = compute_accuracy(all_predictions, all_labels, medium_shot_classes)
    few_shot_acc = compute_accuracy(all_predictions, all_labels, few_shot_classes)

    return many_shot_acc, medium_shot_acc, few_shot_acc
