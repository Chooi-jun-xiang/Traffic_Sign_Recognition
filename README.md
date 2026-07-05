# Traffic Sign Recognition with CNN + Autoencoder + ANN

Image classification of traffic signs using a hybrid deep-learning model built
with **PyTorch**: a CNN feature extractor, an autoencoder for compact feature
encoding, and an ANN classifier head.

## Model Architecture

```
Input (3x224x224)
   │
   ▼
CNN Feature Extractor        3 conv blocks (32→64→128 channels),
   │                         BatchNorm + ReLU + MaxPool
   ▼
Autoencoder Encoder          flattened → 512 → 256 (BatchNorm + ReLU)
   │                         (decoder used for reconstruction loss)
   ▼
ANN Classifier               Dropout → Linear layers → NUM_CLASSES logits
```

## Dataset

- Malaysia road sign dataset (Roboflow, COCO-format annotations), 63 classes
- `split_dataset.py` merges the source splits and re-splits them
  **stratified by class** into train / valid / test (86 / 13 / 1)
- Training uses data augmentation: random horizontal flip, rotation,
  color jitter; all splits resized to 224x224 and ImageNet-normalized

## Evaluation

`evaluate_and_plot.py` reports:

- Accuracy and Top-k accuracy
- F1 score
- Confusion matrix plot
- ROC curves per class

## Project Structure

| Path | Role |
|------|------|
| `main.py` | Training loop, LR scheduling, model saving |
| `models/traffic_sign_ann.py` | CNN + Autoencoder + ANN model definition |
| `datasets/traffic_sign_dataset.py` | COCO-annotation Dataset + label mapping |
| `utils/training.py` | Per-epoch training step |
| `utils/evaluation.py` | Metrics + plots |
| `utils/model_utils.py` | Save / load helpers |
| `split_dataset.py` | Stratified dataset splitting |
| `gui_app.py` | Simple GUI for inference on single images |

## How to Run

```bash
pip install torch torchvision matplotlib scikit-learn pillow tqdm

# 1. Place the COCO dataset export in ./test.v1i.coco and split it
python split_dataset.py

# 2. Train
python main.py

# 3. Evaluate with plots
python evaluate_and_plot.py

# 4. Try the GUI
python gui_app.py
```

## Results

<!-- TODO: fill in after re-running training -->
- Test accuracy: __%
- F1 score: __
