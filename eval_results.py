# eval_results.py — load the saved model and print final test metrics
# (fixes the argument-order bug in the legacy evaluation calls)
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import TrafficSignDataset
from utils.model_utils import load_model
from utils.evaluation import (evaluate_model, get_predictions_and_labels,
                              calculate_topk_accuracy, calculate_f1_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = TrafficSignDataset(
    json_path="data_stratified/test/_annotations.coco.json",
    image_dir="data_stratified/test",
    device=device, transform=tf)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

model = load_model(TrafficSignCNN_AE_ANN, "best_traffic_sign_classifier.pth", device)
model.eval()

test_loss, test_acc = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)

# correct unpacking order: (labels, logits, scores)
all_labels, all_logits, all_scores = get_predictions_and_labels(model, test_loader, device)
all_preds = np.argmax(all_logits, axis=1)

top3 = calculate_topk_accuracy(torch.tensor(all_logits), torch.tensor(all_labels), k=3)
f1 = calculate_f1_score(all_preds, all_labels)

print("\n===== FINAL TEST RESULTS =====")
print(f"Test Accuracy : {test_acc:.2f}%")
print(f"Top-3 Accuracy: {top3:.2f}%")
print(f"F1 (weighted) : {f1:.4f}")
