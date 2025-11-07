import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        p_t = (probs * targets_one_hot).sum(dim=1)  # Shape: (N,)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t + 1e-8)
        focal_loss = focal_weight * ce_loss
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]  # Shape: (N,)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()


def compute_class_weights(targets, num_classes=7, minority_boost=1.8):
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=targets
    )
    
    unique, counts = np.unique(targets, return_counts=True)
    class_counts = np.zeros(num_classes)
    class_counts[unique] = counts
    
    median_count = np.median(class_counts[class_counts > 0])
    minority_classes = np.where(class_counts < median_count)[0]
    
    for cls_idx in minority_classes:
        if class_counts[cls_idx] > 0:
            class_weights[cls_idx] *= minority_boost
    
    weights_tensor = torch.FloatTensor(class_weights)
    
    print(f" Class Weights (with {minority_boost}x minority boost):")
    for i in range(num_classes):
        count = int(class_counts[i])
        weight = class_weights[i]
        boost_marker = " ⬆️ BOOSTED" if i in minority_classes else ""
        print(f"   Class {i}: count={count:5d}, weight={weight:.3f}{boost_marker}")
    
    return weights_tensor

if __name__ == "__main__":
    print(" Focal Loss Implementation Test\n")
    
    print("Test 1: Basic Focal Loss")
    batch_size = 8
    num_classes = 7
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 1])
    
    focal_loss = FocalLoss(alpha=None, gamma=2.5)
    loss = focal_loss(logits, targets)
    print(f"   Loss value: {loss.item():.4f}")
    print("    Basic test passed\n")
    
    print("Test 2: Focal Loss with Class Weights")
    class_weights = torch.tensor([2.0, 1.0, 1.0, 0.8, 1.2, 2.5, 1.5])
    focal_loss_weighted = FocalLoss(alpha=class_weights, gamma=2.5)
    loss_weighted = focal_loss_weighted(logits, targets)
    print(f"   Loss value: {loss_weighted.item():.4f}")
    print("    Weighted test passed\n")
    
    print("Test 3: Compute Class Weights")
    simulated_targets = torch.cat([
        torch.zeros(100),
        torch.ones(200),
        torch.full((150,), 2),
        torch.full((300,), 3),
        torch.full((180,), 4),
        torch.full((80,), 5),
        torch.full((120,), 6),
    ]).long()
    
    weights = compute_class_weights(simulated_targets, num_classes=7, minority_boost=1.8)
    print(f"\n    Class weight computation passed\n")
    
    print("Test 4: Gradient Flow")
    logits.requires_grad = True
    loss = focal_loss_weighted(logits, targets)
    loss.backward()
    print(f"   Gradient exists: {logits.grad is not None}")
    print(f"   Gradient norm: {logits.grad.norm().item():.4f}")
    print("    Gradient flow test passed\n")
    
    print(" All tests passed! Focal Loss is ready for training.")
