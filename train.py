import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MultiScaleAttentionUNet
from data import MultiModalPlumeDataset, MultiModalDataProcessor


class COAwareBCELoss(nn.Module):
    """
    CO-aware weighted BCE.
    CO is min-max normalised per sample. Foreground weights increase with CO,
    background weights increase as CO decreases. The loss is normalised by
    the total weight.
    """

    def __init__(self, co_weight_fg=2.0, co_weight_bg=2.5):
        super(COAwareBCELoss, self).__init__()
        self.co_weight_fg = co_weight_fg
        self.co_weight_bg = co_weight_bg

    def forward(self, logits, targets, co_input):
        co_min = co_input.amin(dim=(2, 3), keepdim=True)
        co_max = co_input.amax(dim=(2, 3), keepdim=True)
        co_norm = ((co_input - co_min) / (co_max - co_min + 1e-6)).clamp(0.0, 1.0)

        fg_weight = 1.0 + (self.co_weight_fg - 1.0) * co_norm
        bg_weight = 1.0 + (self.co_weight_bg - 1.0) * (1.0 - co_norm)
        weight_map = torch.where(targets > 0.5, fg_weight, bg_weight)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce * weight_map).sum() / (weight_map.sum() + 1e-6)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    return ((intersection + 1e-8) / (union + 1e-8)).item()


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_iou = 0.0, 0.0

    with torch.set_grad_enabled(training):
        for co, fire, rgb1, false_rgb, mask in loader:
            co, fire = co.to(device), fire.to(device)
            rgb1, false_rgb = rgb1.to(device), false_rgb.to(device)
            mask = mask.to(device)

            output = model(co, fire, rgb1, false_rgb)
            loss = loss_fn(output, mask, co_input=co)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_iou += calculate_iou(output, mask)

    n_batches = len(loader)
    return total_loss / n_batches, total_iou / n_batches


def load_data(split_dir, fold, batch_size, num_workers):
    """
    Training split of the development set, with the held-out fold
    used only for early stopping. Statistics come from the training split.
    """
    fold_dir = os.path.join(split_dir, f'fold_{fold}')
    train_lists = np.load(os.path.join(fold_dir, 'train_file_lists.npy'), allow_pickle=True).item()
    val_lists = np.load(os.path.join(fold_dir, 'val_file_lists.npy'), allow_pickle=True).item()
    co_stats, fire_stats, rgb_stats = np.load(os.path.join(fold_dir, 'data_stats.npy'), allow_pickle=True)

    processor = MultiModalDataProcessor(image_size=(256, 256))
    processor.co_stats = co_stats
    processor.fire_stats = fire_stats
    processor.rgb_stats = rgb_stats

    train_dataset = MultiModalPlumeDataset(train_lists, processor, augment=True, normalize=True)
    val_dataset = MultiModalPlumeDataset(val_lists, processor, augment=False, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    stats = {'co': co_stats, 'fire': fire_stats, 'rgb': rgb_stats}
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader, stats


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, stats = load_data(
        args.split_dir, args.fold, args.batch_size, args.num_workers
    )

    model = MultiScaleAttentionUNet(
        n_channels_co=1,
        n_channels_fire=1,
        n_channels_rgb=3,
        n_classes=1,
        use_attention=True,
        use_fusion=True,
        use_fire=True,
        use_rgb1=True,
        use_false_rgb=True
    ).to(device)

    loss_fn = COAwareBCELoss(co_weight_fg=args.co_weight_fg, co_weight_bg=args.co_weight_bg)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_iou = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, IoU {train_iou:.4f} | "
              f"val loss {val_loss:.4f}, IoU {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'data_stats': stats,
                'best_val_iou': best_val_iou
            }, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IsoPlume-Net')
    parser.add_argument('--split_dir', required=True,
                        help='Directory with the fold folders from data.py')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fold', type=int, default=4,
                        help='Held-out fold used for early stopping')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--co_weight_fg', type=float, default=2.0)
    parser.add_argument('--co_weight_bg', type=float, default=2.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())