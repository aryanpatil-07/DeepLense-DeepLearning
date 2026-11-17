import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import label_binarize
import time

# -------------------------------------------------------------
# 1. SYNTHETIC GRAVITATIONAL LENS DATA GENERATOR
# (Simulates: 0: No Substructure, 1: Subhalo, 2: Vortex)
# -------------------------------------------------------------
def generate_synthetic_lenses(num_samples=900, seed=42):
    np.random.seed(seed)
    images = np.zeros((num_samples, 1, 64, 64), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    y, x = np.ogrid[-32:32, -32:32]
    radius = np.sqrt(x**2 + y**2)

    for i in range(num_samples):
        label = i % 3
        labels[i] = label

        # Base Einstein Ring (circular light ring)
        ring = np.exp(-((radius - 18) ** 2) / (2 * 3.5**2))

        if label == 1:
            # Subhalo: Add a localized bright clump/distortion
            cx, cy = np.random.randint(-15, 15), np.random.randint(-15, 15)
            clump = 0.8 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 2.5**2))
            ring += clump
        elif label == 2:
            # Vortex / Axion Wave DM: Add subtle wavy ripples across the ring
            ripple = 0.3 * np.sin(x / 2.0) * np.cos(y / 2.0)
            ring += ripple

        # Add Gaussian background noise (telescope sensor noise)
        noise = np.random.normal(0, 0.08, (64, 64))
        images[i, 0] = ring + noise

    return images, labels

class LensDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_dataloaders(batch_size=32, num_samples=900, split_ratio=0.8):
    raw_images, raw_labels = generate_synthetic_lenses(num_samples=num_samples)
    split = int(split_ratio * len(raw_images))

    train_ds = LensDataset(raw_images[:split], raw_labels[:split])
    val_ds = LensDataset(raw_images[split:], raw_labels[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# -------------------------------------------------------------
# 2. MODEL FACTORIES
# -------------------------------------------------------------
def build_resnet18(num_classes=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_coatnet_nano(num_classes=3, img_size=64):
    model = timm.create_model(
        'coatnet_nano_rw_224',
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
        img_size=img_size
    )
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------------------------------------
# 3. TRAINING & EVALUATION ROUTINE
# -------------------------------------------------------------
def train_and_eval(model, train_loader, val_loader, epochs=6, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

    train_time = time.time() - t0

    # Final Validation Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    class_aurocs = []
    fprs, tprs = [], []
    for i in range(3):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        class_aurocs.append(roc_auc)
        fprs.append(fpr)
        tprs.append(tpr)

    macro_auroc = np.mean(class_aurocs)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'acc': acc,
        'class_aurocs': class_aurocs,
        'macro_auroc': macro_auroc,
        'fprs': fprs,
        'tprs': tprs,
        'cm': cm,
        'train_time': train_time
    }

# -------------------------------------------------------------
# 4. EXECUTE BENCHMARK
# -------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing DeepLense Benchmark on device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=32, num_samples=900)

    print("\n--- Training Model 1: ResNet-18 (CNN Baseline) ---")
    resnet = build_resnet18()
    resnet_params = count_parameters(resnet)
    print(f"ResNet-18 Parameters: {resnet_params:,}")
    resnet_res = train_and_eval(resnet, train_loader, val_loader, epochs=6, lr=1e-3, device=device)
    print(f"ResNet-18 Finished in {resnet_res['train_time']:.1f}s | Acc: {resnet_res['acc']:.4f} | Macro AUROC: {resnet_res['macro_auroc']:.4f}")

    print("\n--- Training Model 2: CoAtNet-Nano (Hybrid CNN-Transformer) ---")
    coatnet = build_coatnet_nano()
    coatnet_params = count_parameters(coatnet)
    print(f"CoAtNet-Nano Parameters: {coatnet_params:,}")
    coatnet_res = train_and_eval(coatnet, train_loader, val_loader, epochs=6, lr=5e-4, device=device)
    print(f"CoAtNet-Nano Finished in {coatnet_res['train_time']:.1f}s | Acc: {coatnet_res['acc']:.4f} | Macro AUROC: {coatnet_res['macro_auroc']:.4f}")

    # -------------------------------------------------------------
    # 5. GENERATE COMPARATIVE BENCHMARK VISUALIZATIONS
    # -------------------------------------------------------------
    class_names = ['No-Sub', 'Subhalo (CDM)', 'Vortex (Axion)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Loss Curves
    epochs_range = range(1, 7)
    axes[0, 0].plot(epochs_range, resnet_res['train_losses'], 'b--', label='ResNet-18 Train')
    axes[0, 0].plot(epochs_range, resnet_res['val_losses'], 'b-', label='ResNet-18 Val')
    axes[0, 0].plot(epochs_range, coatnet_res['train_losses'], 'r--', label='CoAtNet-Nano Train')
    axes[0, 0].plot(epochs_range, coatnet_res['val_losses'], 'r-', label='CoAtNet-Nano Val')
    axes[0, 0].set_title('Training & Validation Loss Convergence', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: ROC-AUC Comparison
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(3):
        axes[0, 1].plot(
            resnet_res['fprs'][i], resnet_res['tprs'][i],
            linestyle='--', color=colors[i],
            label=f"{class_names[i]} (ResNet AUC={resnet_res['class_aurocs'][i]:.3f})"
        )
        axes[0, 1].plot(
            coatnet_res['fprs'][i], coatnet_res['tprs'][i],
            linestyle='-', color=colors[i],
            label=f"{class_names[i]} (CoAtNet AUC={coatnet_res['class_aurocs'][i]:.3f})"
        )
    axes[0, 1].plot([0, 1], [0, 1], 'k:', alpha=0.5)
    axes[0, 1].set_title('Multi-Class One-vs-Rest ROC Curves Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(fontsize=8, loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Confusion Matrix ResNet-18
    disp1 = ConfusionMatrixDisplay(confusion_matrix=resnet_res['cm'], display_labels=['No-Sub', 'Subhalo', 'Vortex'])
    disp1.plot(cmap='Blues', ax=axes[1, 0], colorbar=False)
    axes[1, 0].set_title(f"ResNet-18 Confusion Matrix (Acc: {resnet_res['acc']*100:.1f}%)", fontsize=12, fontweight='bold')

    # Plot 4: Confusion Matrix CoAtNet-Nano
    disp2 = ConfusionMatrixDisplay(confusion_matrix=coatnet_res['cm'], display_labels=['No-Sub', 'Subhalo', 'Vortex'])
    disp2.plot(cmap='Purples', ax=axes[1, 1], colorbar=False)
    axes[1, 1].set_title(f"CoAtNet-Nano Confusion Matrix (Acc: {coatnet_res['acc']*100:.1f}%)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_path = r'C:\Users\Admin\DeepLense-DeepLearning\deeplense_benchmark_comparison.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nComparative benchmark plots successfully saved to: {plot_path}")

    # Output stats for README
    print("\n=== SUMMARY BENCHMARK METRICS ===")
    print(f"ResNet-18: Params={resnet_params}, Acc={resnet_res['acc']:.4f}, Macro_AUC={resnet_res['macro_auroc']:.4f}, NoSub_AUC={resnet_res['class_aurocs'][0]:.4f}, Subhalo_AUC={resnet_res['class_aurocs'][1]:.4f}, Vortex_AUC={resnet_res['class_aurocs'][2]:.4f}")
    print(f"CoAtNet-Nano: Params={coatnet_params}, Acc={coatnet_res['acc']:.4f}, Macro_AUC={coatnet_res['macro_auroc']:.4f}, NoSub_AUC={coatnet_res['class_aurocs'][0]:.4f}, Subhalo_AUC={coatnet_res['class_aurocs'][1]:.4f}, Vortex_AUC={coatnet_res['class_aurocs'][2]:.4f}")
