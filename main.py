import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.optim as optim # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torchvision import transforms # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import numpy as np # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from loguru import logger # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import os # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import warnings  # 新增：用來過濾特定 FutureWarning
from network import (
    BaselineCNN,
    BaselineNN,
    CNN_3_128_Homework,
    CNN_3_128,
    ImpNN,
    ResCNN,
    Resnet,
    train_epoch,
    evaluate,
    test
)
from sklearn.metrics import classification_report, confusion_matrix # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import seaborn as sns # pyright: ignore[reportMissingModuleSource, reportMissingImports]
# import KaggleApi # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import subprocess # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from utils import FashionMNISTCSVDataset
from kaggle.api.kaggle_api_extended import KaggleApi # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tqdm.auto import tqdm # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import sys # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torchvision.transforms import functional as TF # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from PIL import ImageFilter
import random

# 過濾掉 DataParallel 內部使用舊版 torch.cuda.amp.autocast 的 FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
)

# Fashion MNIST class names
from utils import (
    CLASS_NAMES,
    CUR_MODEL,
    KAGGLE_SUBMIT,
    EARLY_STOPPING,
    MULTI_GPU,
    NUM_WORKERS,
    BATCH_NORM,
    RESIDUAL,
    DROPOUT,
    DROPOUT_ENABLED,
    POOLING,
    POOLING_ENABLED
)

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Visualize and save a confusion matrix for validation results."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def submit_to_kaggle(file_path: str, competition: str, message: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Submission file not found: {file_path}")
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"[Kaggle] Submitting '{file_path}' to '{competition}' ...")
        api.competition_submit(file_path, message, competition)
        logger.info("[Kaggle] Submission sent.")
    except ImportError:
        # print("[Kaggle] kaggle 套件未安裝，改用 CLI...")
        logger.info("[Kaggle] kaggle package not installed, using CLI instead...")
        cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", file_path, "-m", message]
        subprocess.run(cmd, check=True)
        logger.info("[Kaggle] Submission sent via CLI.")

class Cutout(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class RandomEdge(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.FIND_EDGES)
        return img
    
# In main.py

class PaperAugmentation:
    def __init__(self):
        # 1. Edge Detection (Paper Fig 2d)
        # FIX: RandomEdge must happen on PIL image, BEFORE ToTensor
        self.edge = transforms.Compose([
            RandomEdge(p=1.0),          # <--- Applied to PIL Image
            transforms.ToTensor(),      # <--- Converts to Tensor
            transforms.Normalize((0.2860,), (0.3530,)) 
        ])
        
        # 2. Color Jittering (Paper Fig 2b)
        self.jitter = transforms.Compose([
            transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3)), # <--- Applied to PIL
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        # 3. Geometric: Rotation/Shift (Paper Fig 2e)
        self.geo = transforms.Compose([
            transforms.RandomAffine(degrees=8, translate=(2/28, 2/28)), # <--- Applied to PIL
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        # 4. Flipping (Paper Fig 2c)
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0), # <--- Applied to PIL
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        # 5. Original (Paper Fig 2a)
        self.orig = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

    def __call__(self, img):
        # Input 'img' is a PIL Image from the Dataset
        choice = random.random()
        if choice < 0.2:
            return self.edge(img)
        elif choice < 0.4:
            return self.jitter(img)
        elif choice < 0.6:
            return self.geo(img)
        elif choice < 0.8:
            return self.flip(img)
        else:
            return self.orig(img)

# In main.py, use it like this:
# train_transform = PaperAugmentation()

def main():
    global logger
    # Set random seed for reproducibility
    torch.manual_seed(45)
    np.random.seed(45)

    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()  # Remove default logger

    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", colorize=True)

    log_name = f'{CUR_MODEL}'
    if EARLY_STOPPING:
        log_name += '_ES'
    if MULTI_GPU:
        log_name += '_MGPU'
    if BATCH_NORM:
        log_name += '_BN'
    if RESIDUAL:
        log_name += '_RES'
    if DROPOUT_ENABLED and DROPOUT > 0:
        log_name += f'_D{DROPOUT}'
    if POOLING_ENABLED and POOLING:
        log_name += f'_P{POOLING}'

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    plot_name = f'{log_name}_training_history_{today}.png'

    logger.add(
        os.path.join(output_dir, f'{log_name}_{today}.log'),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
        colorize=False
    )

    logger = logger.opt(colors=True)
    
    # Hyperparameters
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-3
    dropout_rate = DROPOUT
    early_stopping_patience = 20
    
    # Model selection: 'CNN', 'ResNet10', 'ResNet18'
    model_type = CUR_MODEL
    
    # Device configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            device = torch.device(f'cuda:{num_gpus - 1}')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Cofiguration logging
    if EARLY_STOPPING:
        # logger.info("Early Stopping: Enabled") # green for enabled, blue for disabled
        logger.info(f"Early Stopping: <green>Enabled</green> with <yellow>{early_stopping_patience} </yellow>patience")
    else:
        logger.info("Early Stopping: <blue>Disabled</blue>")
    if MULTI_GPU:
        # logger.info("Multi-GPU: Enabled") # green for enabled, blue for disabled
        logger.info("Multi-GPU: <green>Enabled</green>")
    else:
        logger.info("Multi-GPU: <blue>Disabled</blue>")
    
    if BATCH_NORM:
        logger.info("Batch Normalization: <green>Enabled</green>")
    else:
        logger.info("Batch Normalization: <blue>Disabled</blue>")
    if RESIDUAL:
        logger.info("Residual Connections: <green>Enabled</green>")
    else:
        logger.info("Residual Connections: <blue>Disabled</blue>")

    if DROPOUT_ENABLED and DROPOUT > 0:
        logger.info(f"Dropout: <green>Enabled (Rate: {dropout_rate})</green>")
    else:
        logger.info("Dropout: <blue>Disabled</blue>")
    
    if POOLING_ENABLED and POOLING:
        logger.info(f"Pooling: <green>Enabled ({POOLING} Pooling)</green>")
    else:
        logger.info("Pooling: <blue>Disabled</blue>")

    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Number of Epochs: {num_epochs}")
    logger.info(f"Learning Rate: {learning_rate}")
    
    # Exact Fashion-MNIST statistics
    # Mean: 0.2860, Std: 0.3530
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(15),
    #     transforms.RandomAffine(
    #         degrees=0, 
    #         translate=(0.1, 0.1),
    #         scale=(0.9, 1.1),
    #         shear=5
    #     ),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.2860,), (0.3530,)),
    #     Cutout(n_holes=1, length=8)
    # ])

    # main.py 裡，替換現在的 train_transform

    # train_transform = transforms.Compose([
    #     # 1. Color Jittering：brightness / contrast in (0.7, 1.3)
    #     transforms.ColorJitter(
    #         brightness=(0.7, 1.3),
    #         contrast=(0.7, 1.3)
    #     ),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomAffine(
    #         degrees=8,
    #         translate=(2/28, 2/28),
    #     ),

    #     transforms.ToTensor(),
    #     transforms.Normalize((0.2860,), (0.3530,)),
    # ])

    # train_transform = transforms.Compose([
    #     RandomEdge(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.ColorJitter(
    #         brightness=(0.7, 1.3),
    #         contrast=(0.7, 1.3),
    #     ),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomAffine(
    #         degrees=8,
    #         translate=(2/28, 2/28),
    #         scale=None,
    #         shear=None
    #     ),
    #     transforms.RandomErasing(
    #         p=0.25,
    #         scale=(0.02, 0.2),
    #         ratio=(0.3, 3.3),
    #         value=0.0
    #     ),
    #     transforms.Normalize((0.2860,), (0.3530,))
    # ])

    train_transform = PaperAugmentation()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load Fashion MNIST dataset from CSV files
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test4students.csv')
    
    logger.info(f"Loading Fashion MNIST dataset from CSV files in {data_dir}")
    
    # Load full training dataset
    full_train_dataset = FashionMNISTCSVDataset(
        csv_file=train_csv,
        transform=train_transform,
        has_idx=False
    )
    
    # Load test dataset
    test_dataset = FashionMNISTCSVDataset(
        csv_file=test_csv,
        transform=transform,
        has_idx=True,
        is_test=True
    )
    
    # Split training data into train and validation
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS
        )
    # Create model
    logger.info(f"Creating {model_type} model")

    if (model_type == 'BaseCNN'):
        model = BaselineCNN(num_classes=10).to(device)
    elif (model_type == 'BaseNN'):
        model = BaselineNN(num_classes=10).to(device)
    elif (model_type == 'CNN'):
        model = CNN_3_128_Homework(num_classes=10).to(device)
    elif (model_type == 'CNN_3_128'):
        model = CNN_3_128(num_classes=10).to(device)
    elif (model_type == 'ImpNN'):
        model = ImpNN(num_classes=10).to(device)
    elif (model_type == 'Resnet'):
        model = Resnet(num_classes=10).to(device)
    elif (model_type == 'ResCNN'):
        model = ResCNN(num_classes=10).to(device)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return
    
    if MULTI_GPU and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # # Switch to SGD with Nesterov Momentum
    # # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # # Switch to CosineAnnealing (Smoother decay than Plateau)
    # # T_max should be equal to num_epochs
    # # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # optimizer = optim.AdamW(
    #     model.parameters(), 
    #     lr=learning_rate,
    #     weight_decay=1e-4,
    #     betas=(0.9, 0.999)
    # )
    
    # # Learning rate scheduler
    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # # 在 main.py 中調整
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.005,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.2,
    #     anneal_strategy='cos',
    #     div_factor=25.0,
    #     final_div_factor=1e4 
    # )

    criterion = nn.CrossEntropyLoss()      # 先關掉 label_smoothing
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,                           # 論文 lr
        weight_decay=0.0                   # 先不用 weight decay（看結果再開）
    )

    # 簡單一點，用 ReduceLROnPlateau + early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=20
    )

    # some settings to logger
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    early_stopping_counter = 0

    # Training loop
    for epoch in range(num_epochs):

        train_loss, train_acc = train_epoch(
            model,
            tqdm(train_loader, desc=f"Train {epoch}", leave=False),
            criterion,
            optimizer,
            device,
        )
        
        val_loss, val_acc = evaluate(
            model,
            tqdm(val_loader, desc=f"Validation {epoch}", leave=False),
            criterion,
            device,
        )

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        # scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log progress
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: <yellow>{train_loss:.4f}</yellow>, Train Acc: <yellow>{train_acc:.2f}%</yellow>")
        logger.info(f"  Val Loss: <yellow>{val_loss:.4f}</yellow>, Val Acc: <yellow>{val_acc:.2f}%</yellow>")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, f'best_{log_name}_{today}.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  Best model saved! Val Acc: <yellow>{val_acc:.2f}%</yellow>")
        
            if EARLY_STOPPING:
                early_stopping_counter = 0
        # Plot training history every 5 epochs
        else:
            if EARLY_STOPPING:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                    break
        logger.info(f"  Early Stopping Counter: {early_stopping_counter}/{early_stopping_patience}")
        if (epoch + 1) % 5 == 0:
            plot_training_history(
                train_losses, train_accs, val_losses, val_accs,
                os.path.join(output_dir, plot_name)
            )

    # Parameter Counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    
    # Final training history plot
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(output_dir, plot_name)
    )
    
    # Load best model and evaluate on test set
    logger.info("Loading best model for final evaluation")
    best_model_path = os.path.join(output_dir, f'best_{log_name}_{today}.pth')
    state_dict = torch.load(
        best_model_path,
        weights_only=True,
        map_location=device
    )
    model.load_state_dict(state_dict)

    # Re-run validation to capture predictions for confusion matrix visualization
    val_loss, val_acc, val_labels, val_preds = evaluate(
        model,
        tqdm(val_loader, desc="Validation (Confusion Matrix)", leave=False),
        criterion,
        device,
        return_predictions=True
    )
    logger.info(f"Final Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    cm_path = os.path.join(output_dir, f'{log_name}_confusion_matrix_{today}.png')
    plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, cm_path)
    val_report = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info("Validation Classification Report:\n" + val_report)

    # test_pred = test(model, test_loader, device)
    test_pred = test(model, tqdm(test_loader, desc="Testing", leave=False), device, n_tta=10)
    # save test predictions to test_results.csv
    test_results_path = os.path.join(output_dir, 'pred.csv')
    with open(test_results_path, 'w') as f:
        f.write('idx,label\n')
        for idx, label in test_pred:
            f.write(f'{idx},{label}\n')
    logger.info(f"Test predictions saved to {test_results_path}")

    # upload test_results.csv to kaggle to get test accuracy
    # kaggle competitions submit -c hw-5-comparing-between-nn-cnn-using-fashion-mnist -f ./test_results.csv -m "{date-time}"
    
    if KAGGLE_SUBMIT:
        try:
            import datetime
            Datetime = datetime.datetime
            msg = f"{model_type} at {Datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if EARLY_STOPPING:
                msg += " with Early Stopping"
            if MULTI_GPU:
                msg += " using Multi-GPU"
            if BATCH_NORM:
                msg += " with BatchNorm"
            if RESIDUAL:
                msg += " with Residuals"
            if DROPOUT:
                msg += f" with Dropout({dropout_rate})"
            if POOLING:
                msg += f" with {POOLING} Pooling"
            submit_to_kaggle(os.path.join(output_dir, 'pred.csv'), "hw-5-fashion-mnist-new", msg)
        except Exception as e:
            print(f"[Kaggle] Submission failed: {e}")
    

if __name__ == '__main__':
    main()
