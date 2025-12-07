import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from typing import Tuple # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from loguru import logger # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.amp import autocast, GradScaler  # pyright: ignore[reportMissingModuleSource, reportMissingImports]

from utils import (
    BATCH_NORM, DROPOUT_ENABLED, DROPOUT, RESIDUAL, POOLING, POOLING_ENABLED
)
    
# Resnet-like CNN
class ResBlockPaper(nn.Module):
    def __init__(self, in_channels, out_channels, use_projection=False):
        super().__init__()
        # Paper uses 5x5 kernels 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Paper mentions Dropout 0.23 AFTER each block [cite: 202]
        self.dropout = nn.Dropout2d(p=DROPOUT) if DROPOUT_ENABLED else nn.Identity()

        self.use_projection = use_projection
        # Paper: Shortcut consists of BN followed by Conv 1x1 [cite: 197]
        if self.use_projection:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.proj = None

    def forward(self, x):
        identity = x

        # Note: Paper says BN is applied BEFORE convolution in some tests, 
        # but standard ResNet is Conv-BN-ReLU. 
        # Based on Table VII, we keep standard order but ensure 5x5 kernels.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_projection:
            identity = self.proj(identity)

        if RESIDUAL:
            out += identity

        out = F.relu(out)
        out = self.dropout(out)
        return out

class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # --- Stem ---
        # Paper does not explicitly mention a stem, but usually starts with Conv.
        # Based on Table VII, ResCNN starts immediately with Residual Blocks 
        # or has a Conv2D-1 (64) before. Let's assume an initial 64 conv.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- Residual Blocks ---
        # Block 1: 64 filters, 5x5, No shortcut projection 
        self.block1 = ResBlockPaper(64, 64, use_projection=False)
        
        # Block 2: 64 filters, 5x5, With shortcut projection [cite: 197]
        self.block2 = ResBlockPaper(64, 64, use_projection=True)

        # --- Pooling ---
        # "Two Max-Pooling layers with stride 2... before fully connected" [cite: 203]
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # 28 -> 14
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14 -> 7
        )

        # --- Classifier ---
        # Flatten size: 64 channels * 7 * 7 = 3136
        self.flatten_size = 64 * 7 * 7 
        
        # Dense layers from Table VII 
        # Activation: GeLU for dense layers 
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128) if BATCH_NORM else nn.Identity(),
            nn.GELU(),
            nn.Dropout(p=DROPOUT) if DROPOUT_ENABLED else nn.Identity(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64) if BATCH_NORM else nn.Identity(),
            nn.GELU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32) if BATCH_NORM else nn.Identity(),
            nn.GELU(),
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        
        # CRITICAL FIX: Flatten instead of Global Pool
        x = x.view(x.size(0), -1) 
        
        x = self.fc_layers(x)
        return x
    
# Copy to network.py
class ResCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Paper: 5x5 filters, 64 channels everywhere
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Block 1
        self.block1_conv1 = nn.Conv2d(64, 64, 5, padding=2)
        self.block1_bn1 = nn.BatchNorm2d(64)
        self.block1_conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.block1_bn2 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(p=0.23)
        
        # Block 2 (With Shortcut)
        self.block2_conv1 = nn.Conv2d(64, 64, 5, padding=2)
        self.block2_bn1 = nn.BatchNorm2d(64)
        self.block2_conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.block2_bn2 = nn.BatchNorm2d(64)
        # Shortcut projection (1x1 conv)
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 1)
        )
        self.drop2 = nn.Dropout2d(p=0.23)
        
        # Pooling & Flatten
        self.pool = nn.MaxPool2d(2, 2) # Applied twice
        self.flatten_dim = 64 * 7 * 7 # 3136 features
        
        # Dense Layers (Paper Table VII)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.act_dense = nn.GELU() # Paper uses GeLU for dense
        self.drop_fc = nn.Dropout(p=0.35)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # ResBlock 1 (Simple addition if dims match, paper says no projection here)
        identity = x
        out = F.relu(self.block1_bn1(self.block1_conv1(x)))
        out = self.block1_bn2(self.block1_conv2(out))
        out += identity # Residual
        out = F.relu(out)
        x = self.drop1(out)
        
        # ResBlock 2 (With projection shortcut)
        identity = x
        out = F.relu(self.block2_bn1(self.block2_conv1(x)))
        out = self.block2_bn2(self.block2_conv2(out))
        # Shortcut path
        identity = self.shortcut(identity)
        out += identity # Residual
        out = F.relu(out)
        x = self.drop2(out)
        
        # Pooling (Paper says "two Max-Pooling layers")
        x = self.pool(x) # 28->14
        x = self.pool(x) # 14->7
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense
        x = self.act_dense(self.fc1(x))
        x = self.drop_fc(x)
        x = self.act_dense(self.fc2(x))
        x = self.act_dense(self.fc3(x))
        x = self.fc4(x)
        return x

    
# Current improvement version of cnn-3-128
class CNN_3_128(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1: 1 -> 128 (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Conv2d(1, 64, kernel_size=1) if RESIDUAL else None
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(p=DROPOUT) if DROPOUT_ENABLED else nn.Identity()
        
        # Block 2: 128 -> 128 (14x14 -> 7x7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1) if RESIDUAL else None
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(p=DROPOUT) if DROPOUT_ENABLED else nn.Identity()
        
        # Block 3: 128 -> 128 (7x7, no pooling)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(p=DROPOUT) if DROPOUT_ENABLED else nn.Identity()
        
        # Classifier
        # self.fc1 = nn.Linear(128 * 7 * 7, 512)
        # self.bn_fc = nn.BatchNorm1d(512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout_fc = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if RESIDUAL:
            x = x + self.shortcut1(identity)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        identity = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        if RESIDUAL:
            x = x + self.shortcut2(identity)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        identity = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        if RESIDUAL:
            x = x + identity
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Classifier
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.bn_fc(self.fc1(x)))
        # x = self.dropout_fc(x)
        # return self.fc2(x)
        x= self.global_pool(x)
        x = x.flatten(1)
        x = self.fc2(x)
        return x

    
# ==============================================================================
# CNN-3-128 Homework Edition (High Accuracy + Efficiency Bonus)
# ==============================================================================
class CNN_3_128_Homework(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def make_block(in_c, out_c):
            layers = []
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False))
            if BATCH_NORM: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False))
            if BATCH_NORM: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.block1 = make_block(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=DROPOUT)

        self.block2 = make_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=DROPOUT)

        self.block3 = make_block(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=DROPOUT)

        self.res1 = nn.Conv2d(1, 64, kernel_size=1)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1)
        self.res3 = nn.Sequential()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.block3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x
    
# ==============================================================================
# Baseline NN
# ==============================================================================
# Input → Linear → ReLU → Linear → ReLU → Linear → Softmax (in loss)
class BaselineNN(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        if BATCH_NORM:
            self.bn1 = nn.BatchNorm1d(256)
        if DROPOUT_ENABLED and DROPOUT > 0:
            self.dropout = nn.Dropout(p=DROPOUT)
        self.fc2 = nn.Linear(256, 128)
        if BATCH_NORM:
            self.bn2 = nn.BatchNorm1d(128)
        if DROPOUT_ENABLED and DROPOUT > 0:
            self.dropout = nn.Dropout(p=DROPOUT)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        if BATCH_NORM:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = F.relu(x)
        if DROPOUT_ENABLED and DROPOUT > 0:
            x = self.dropout(x)
        if BATCH_NORM:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = F.relu(x)
        if DROPOUT_ENABLED and DROPOUT > 0:
            x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc3(x)

class ImpNN(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        return self.fc3(x)

# ==============================================================================
# Baseline CNN
# ==============================================================================
# [Conv(3×3, 32)] → ReLU → [Conv(3×3, 64)] → ReLU → MaxPool(2×2) → Flatten → Linear → ReLU → Linear → Softmax (in loss)
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # 預設 spatial dimension (Fashion MNIST 是 28x28)
        # 經過兩次 padding=1 的 3x3 conv，大小依然是 28x28
        spatial_dim = 28 

        if POOLING_ENABLED:
            if POOLING == 'max':
                self.pool  = nn.MaxPool2d(2, 2)
                spatial_dim = 14 # 28 / 2
            elif POOLING == 'avg':
                self.pool  = nn.AvgPool2d(2, 2)
                spatial_dim = 14 # 28 / 2
            elif POOLING == 'none':
                self.pool  = nn.Identity()
                # spatial_dim 保持 28
            elif POOLING == 'str2conv':
                self.pool  = nn.Conv2d(64, 64, kernel_size=2, stride=2)
                spatial_dim = 14 # 28 / 2
            else:
                raise ValueError(f"Unsupported POOLING type: {POOLING}")
        
        # 動態計算 flatten size
        self.flatten_size = 64 * spatial_dim * spatial_dim
        
        # 使用計算後的 size 初始化 fc1
        self.fc1   = nn.Linear(self.flatten_size, 128)
        self.fc2   = nn.Linear(128, num_classes)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

        if RESIDUAL:
            self.residual_shortcut = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)

        # Dropout layers
        if DROPOUT_ENABLED and DROPOUT > 0:
            self.dropout_conv = nn.Dropout2d(p=DROPOUT)
            self.dropout_fc = nn.Dropout(p=DROPOUT)
        else:
            self.dropout_conv = None
            self.dropout_fc = None

    def forward(self, x):
        # --- Block 1 ---
        x = self.conv1(x)
        if BATCH_NORM:
            x = self.batch_norm1(x)
        x = F.relu(x) # ReLU usually comes after BN

        if self.dropout_conv is not None:
            x = self.dropout_conv(x)
            
        # --- Block 2 (Potential Residual Block) ---
        identity = x  # Save the input (32 channels) for the skip connection

        out = self.conv2(x) # Main path (becomes 64 channels)
        
        if BATCH_NORM:
            out = self.batch_norm2(out)

        # Start of Residual Logic
        if RESIDUAL:
            # We must transform identity (32ch) to match out (64ch)
            identity_transformed = self.residual_shortcut(identity)
            out += identity_transformed  # Element-wise addition
        # End of Residual Logic

        x = F.relu(out) # ReLU applies to the combined result

        if self.dropout_conv is not None:
            x = self.dropout_conv(x)
            
        # --- Pooling and FC ---
        if POOLING_ENABLED:
            x = self.pool(x)
            
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        if self.dropout_fc is not None:
            x = self.dropout_fc(x)
            
        return self.fc2(x)

# ==============================================================================
# Training and Evaluation Functions
# ==============================================================================
_scaler = GradScaler()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = (device.type == "cuda")
    
    scaler = GradScaler() if use_amp else None
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device, return_predictions: bool = False):
    """Evaluate the model and optionally return labels/predictions for analysis."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = [] if return_predictions else None
    all_preds = [] if return_predictions else None

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if return_predictions:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    if return_predictions:
        return avg_loss, accuracy, all_labels, all_preds
    return avg_loss, accuracy

# def test(model, test_loader, device):
#     """Test the model and return predictions"""
#     model.eval()
#     all_predictions = []
#     # all_labels = []

#     # There's no labels in the test set, so return predicted labels with indices
#     with torch.no_grad():
#         for images, indices in test_loader:
#             images = images.to(device)
#             outputs = model(images)
#             _, predicted = outputs.max(1)
            
#             for idx, label in zip(indices.numpy(), predicted.cpu().numpy()):
#                 all_predictions.append((idx, label))
#     return all_predictions

def test(model, test_loader, device, n_tta=5):
    """Test with Test Time Augmentation"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for images, indices in test_loader:
            images = images.to(device)
            
            tta_outputs = []
            for _ in range(n_tta):
                outputs = model(images)
                tta_outputs.append(F.softmax(outputs, dim=1))
                
            avg_output = torch.stack(tta_outputs).mean(dim=0)
            _, predicted = avg_output.max(1)
            
            for idx, label in zip(indices.numpy(), predicted.cpu().numpy()):
                all_predictions.append((idx, label))
    
    return all_predictions
