from dataset import get_dataloader
import torch
import torch.nn as nn
from torchvision import models
import copy
def build_model(num_classes=7,freeze_until_block=9):
  model=models.mobilenet_v3_small(weights="IMAGENET1K_V1")
  model.classifier[3]=nn.Linear(model.classifier[3].in_features,num_classes)
  for name,param in model.features.named_parameters():
    block_idx=int(name.split(".")[0])
    param.requires_grad=block_idx>=freeze_until_block
  def freeze_bn_in_frozen(module,block_idx):
    if isinstance(module,nn.BatchNorm2d) and block_idx<freeze_until_block:
      module.eval()
      for p in module.parameters():
        p.requires_grad=False
  for name,module in model.features.named_modules():
    if name=="":
      continue
    top_level_idx=int(name.split(".")[0])
    freeze_bn_in_frozen(module,top_level_idx)
  return model
def print_trainable_summary(model):
  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
  total=sum(p.numel() for p in model.parameters())
  print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")
def build_optimizer(model,backbone_lr=1e-5,head_lr=1e-3):
  backbone_params=[p for p in model.features.parameters() if p.requires_grad]
  head_params=model.classifier.parameters()
  optimizer=torch.optim.Adam([
      {"params":backbone_params,"lr":backbone_lr},
      {"params":head_params,"lr":head_lr}
  ])
  return optimizer
def build_criterion():
  return nn.CrossEntropyLoss()
def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train() if train_mode else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train_mode)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total
def train_model(model, train_loader, val_loader, criterion, optimizer,
                 device, num_epochs=30, patience=10):
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"Best val accuracy: {best_val_acc:.4f}")
    return model, history
def evaluate_model(model, loader, device, class_names, plot_cm=True):
  from sklearn.metrics import (
      accuracy_score, precision_recall_fscore_support,
      classification_report, confusion_matrix
  )

  model.eval()
  all_preds, all_labels = [], []

  with torch.no_grad():
      for images, labels in loader:
          images = images.to(device)
          outputs = model(images)
          preds = outputs.argmax(dim=1).cpu()
          all_preds.extend(preds.tolist())
          all_labels.extend(labels.tolist())
  acc = accuracy_score(all_labels, all_preds)
  print(f"Overall Accuracy: {acc:.4f}\n")
  precision, recall, f1, _ = precision_recall_fscore_support(
      all_labels, all_preds, average="macro", zero_division=0
  )
  print(f"Macro Precision: {precision:.4f}")
  print(f"Macro Recall:    {recall:.4f}")
  print(f"Macro F1-score:  {f1:.4f}\n")
  print("Per-class report:")
  print(classification_report(
      all_labels, all_preds, target_names=class_names, zero_division=0
  ))
  cm = confusion_matrix(all_labels, all_preds)
  print("Confusion matrix (rows = true label, cols = predicted label):")
  print(cm)
  per_class_acc = cm.diagonal() / cm.sum(axis=1)
  print("\nPer-class accuracy:")
  for cls_name, cls_acc in zip(class_names, per_class_acc):
      print(f"  {cls_name:8s}: {cls_acc:.4f}")

  if plot_cm:
      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(figsize=(7, 6))
      im = ax.imshow(cm, cmap="Blues")
      ax.set_xticks(range(len(class_names)))
      ax.set_yticks(range(len(class_names)))
      ax.set_xticklabels(class_names, rotation=45, ha="right")
      ax.set_yticklabels(class_names)
      ax.set_xlabel("Predicted label")
      ax.set_ylabel("True label")
      ax.set_title("Confusion Matrix")
      for i in range(cm.shape[0]):
          for j in range(cm.shape[1]):
              ax.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
      fig.colorbar(im, ax=ax)
      fig.tight_layout()
      plt.show()

  return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(class_names, per_class_acc)),
  }
train_dataloader,val_dataloader=get_dataloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = build_model(num_classes=7, freeze_until_block=9).to(device)
print_trainable_summary(model)
optimizer = build_optimizer(model, backbone_lr=1e-5, head_lr=1e-3)
criterion = build_criterion()
model, history = train_model(model, train_dataloader, val_dataloader, criterion,
                                  optimizer, device, num_epochs=15, patience=4)
# class_names = train_dataset.classes  # e.g. ['akiec','bcc','bkl','df','mel','nv','vasc']
# metrics = evaluate_model(model, val_dataloader, device, class_names)