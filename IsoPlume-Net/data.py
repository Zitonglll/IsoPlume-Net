import os
import re
import random
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiModalDataProcessor:
    """Loads and normalises the CO, fire, RGB and mask inputs."""

    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.co_stats = {'mean': 0.5, 'std': 0.2}
        self.fire_stats = {'mean': 0.1, 'std': 0.3}
        self.rgb_stats = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    def load_co_data(self, path):
        co_image = Image.open(path).convert("L")
        co_image = co_image.resize(self.image_size, Image.BILINEAR)
        co_array = np.array(co_image).astype(np.float32) / 255.0
        return np.nan_to_num(co_array, nan=0.0, posinf=1.0, neginf=0.0)

    def load_fire_data(self, path):
        fire_image = Image.open(path).convert("L")
        fire_image = fire_image.resize(self.image_size, Image.NEAREST)
        fire_array = np.array(fire_image).astype(np.float32) / 255.0
        return (fire_array > 0.5).astype(np.float32)

    def load_rgb_data(self, path):
        rgb_image = Image.open(path).convert("RGB")
        rgb_image = rgb_image.resize(self.image_size, Image.BILINEAR)
        return np.array(rgb_image).astype(np.float32) / 255.0

    def load_mask(self, path):
        mask_image = Image.open(path).convert("L")
        mask_image = mask_image.resize(self.image_size, Image.NEAREST)
        mask = np.array(mask_image).astype(np.float32) / 255.0
        return (mask > 0.5).astype(np.float32)

    def compute_dataset_statistics(self, file_lists, sample_size=50, seed=42):
        """Normalisation statistics computed from the training split only."""
        rng = np.random.RandomState(seed)
        n_samples = min(sample_size, len(file_lists['co']))
        indices = rng.choice(len(file_lists['co']), n_samples, replace=False)

        co_data, fire_data, rgb_data = [], [], []
        for idx in indices:
            co_data.append(self.load_co_data(file_lists['co'][idx]))
            fire_data.append(self.load_fire_data(file_lists['fire'][idx]))
            rgb_data.append(self.load_rgb_data(file_lists['rgb1'][idx]))
            rgb_data.append(self.load_rgb_data(file_lists['false_rgb'][idx]))

        co_array = np.stack(co_data)
        self.co_stats['mean'] = np.mean(co_array)
        self.co_stats['std'] = np.std(co_array)

        fire_array = np.stack(fire_data)
        self.fire_stats['mean'] = np.mean(fire_array)
        self.fire_stats['std'] = np.std(fire_array)

        rgb_array = np.stack(rgb_data)
        self.rgb_stats['mean'] = np.mean(rgb_array, axis=(0, 1, 2)).tolist()
        self.rgb_stats['std'] = np.std(rgb_array, axis=(0, 1, 2)).tolist()

        return self.co_stats, self.fire_stats, self.rgb_stats


class MultiModalPlumeDataset(Dataset):
    """
    file_lists: dict with aligned lists of PNG paths for the keys
        'co', 'fire', 'rgb1', 'false_rgb' and 'mask'.
    """

    def __init__(self, file_lists, processor=None, augment=False,
                 normalize=True, cache_data=False):
        self.file_lists = file_lists
        self.processor = processor or MultiModalDataProcessor()
        self.augment = augment
        self.normalize = normalize
        self.cache_data = cache_data

        lengths = {key: len(files) for key, files in self.file_lists.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Data misalignment: {lengths}")
        self.length = list(lengths.values())[0]

        if cache_data:
            self.cached_data = [self._load_sample(idx) for idx in range(self.length)]

    def _load_sample(self, idx):
        return {
            'co': self.processor.load_co_data(self.file_lists['co'][idx]),
            'fire': self.processor.load_fire_data(self.file_lists['fire'][idx]),
            'rgb1': self.processor.load_rgb_data(self.file_lists['rgb1'][idx]),
            'false_rgb': self.processor.load_rgb_data(self.file_lists['false_rgb'][idx]),
            'mask': self.processor.load_mask(self.file_lists['mask'][idx])
        }

    def __len__(self):
        return self.length

    def apply_augmentation(self, sample):
        """Flips, rotation, random crop with rescaling, and colour jitter on RGB only."""
        if not self.augment:
            return sample

        if random.random() > 0.5:
            for key in sample:
                sample[key] = np.fliplr(sample[key]).copy()

        if random.random() > 0.5:
            for key in sample:
                sample[key] = np.flipud(sample[key]).copy()

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            for key in sample:
                sample[key] = self._rotate_image(
                    sample[key], angle, nearest=(key in ['fire', 'mask'])
                )

        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.0)
            h, w = sample['co'].shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h) if h > new_h else 0
            left = random.randint(0, w - new_w) if w > new_w else 0

            for key in sample:
                cropped = sample[key][top:top + new_h, left:left + new_w]
                interpolation = cv2.INTER_NEAREST if key in ['fire', 'mask'] else cv2.INTER_LINEAR
                sample[key] = cv2.resize(cropped, (w, h), interpolation=interpolation)

        if random.random() > 0.5:
            for key in ['rgb1', 'false_rgb']:
                sample[key] = self._color_jitter(sample[key])

        return sample

    def _rotate_image(self, image, angle, nearest=False):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpAffine(image, M, (w, h), flags=interpolation)

    def _color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2):
        if random.random() > 0.5:
            factor = 1 + random.uniform(-brightness, brightness)
            image = np.clip(image * factor, 0, 1)

        if random.random() > 0.5:
            factor = 1 + random.uniform(-contrast, contrast)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 1)

        if random.random() > 0.5:
            factor = 1 + random.uniform(-saturation, saturation)
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])[..., None]
            image = np.clip(gray + factor * (image - gray), 0, 1)

        return image

    def normalize_data(self, sample):
        if not self.normalize:
            return sample

        p = self.processor
        sample['co'] = (sample['co'] - p.co_stats['mean']) / (p.co_stats['std'] + 1e-8)
        sample['fire'] = (sample['fire'] - p.fire_stats['mean']) / (p.fire_stats['std'] + 1e-8)

        for key in ['rgb1', 'false_rgb']:
            mean = np.array(p.rgb_stats['mean'], dtype=np.float32)
            std = np.array(p.rgb_stats['std'], dtype=np.float32)
            sample[key] = (sample[key] - mean) / (std + 1e-8)

        return sample

    def __getitem__(self, idx):
        if self.cache_data:
            sample = {key: value.copy() for key, value in self.cached_data[idx].items()}
        else:
            sample = self._load_sample(idx)

        sample = self.apply_augmentation(sample)
        sample = self.normalize_data(sample)

        co_tensor = torch.tensor(sample['co'], dtype=torch.float32).unsqueeze(0)
        fire_tensor = torch.tensor(sample['fire'], dtype=torch.float32).unsqueeze(0)
        rgb1_tensor = torch.tensor(sample['rgb1'], dtype=torch.float32).permute(2, 0, 1)
        false_rgb_tensor = torch.tensor(sample['false_rgb'], dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(sample['mask'], dtype=torch.float32).unsqueeze(0)

        return co_tensor, fire_tensor, rgb1_tensor, false_rgb_tensor, mask_tensor


def extract_key_identifier(filename):
    """
    Extract the sample identifier YYYYDDD_HHMM_lat_lon from a file name.
    """
    match = re.search(r'(\d{7}_\d{4}_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?)', filename)
    return match.group(1) if match else None


def create_file_lists(data_folders):
    """Align the files of all modalities by their sample identifier."""
    mappings = {}
    for key, folder in data_folders.items():
        if not os.path.exists(folder):
            raise ValueError(f"Folder not found: {folder}")
        mappings[key] = {}
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.png'):
                continue
            identifier = extract_key_identifier(f)
            if identifier is None:
                raise ValueError(f"No identifier found in {f}")
            mappings[key][identifier] = os.path.join(folder, f)

    common_ids = sorted(set.intersection(*[set(m) for m in mappings.values()]))
    if not common_ids:
        raise ValueError("No aligned samples found across data folders")

    print(f"Aligned samples: {len(common_ids)}")
    return {key: [mappings[key][i] for i in common_ids] for key in mappings}


def build_groups(identifiers, window_deg=4.0):
    """
    Group samples that share the S5P acquisition day and have overlapping
    4 x 4 degree input windows, as connected components.
    """
    parsed = []
    for identifier in identifiers:
        day, _, lat, lon = identifier.split('_')
        parsed.append((day, float(lat), float(lon)))

    n = len(parsed)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if parsed[a][0] != parsed[b][0]:
                continue
            dlat = abs(parsed[a][1] - parsed[b][1])
            dlon = abs(parsed[a][2] - parsed[b][2])
            dlon = min(dlon, 360.0 - dlon)
            if dlat < window_deg and dlon < window_deg:
                parent[find(b)] = find(a)

    return np.array([find(i) for i in range(n)])


def prepare_k_fold_data(file_lists, output_dir, k=5, n_test=20, seed=42):
    """
    Split into an independent test set and K folds of the development set.
    """
    os.makedirs(output_dir, exist_ok=True)

    identifiers = [extract_key_identifier(os.path.basename(p)) for p in file_lists['co']]
    groups = build_groups(identifiers)

    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    test_indices = []
    for g in unique_groups:
        if len(test_indices) >= n_test:
            break
        test_indices.extend(np.where(groups == g)[0].tolist())
    test_groups = set(groups[test_indices])
    dev_indices = [i for i in range(len(groups)) if groups[i] not in test_groups]

    print(f"Development: {len(dev_indices)}, Test: {len(test_indices)}")

    test_lists = {key: [file_lists[key][i] for i in test_indices] for key in file_lists}
    np.save(os.path.join(output_dir, 'test_file_lists.npy'), test_lists)

    fold_sizes = [0] * k
    fold_of_group = {}
    for g in unique_groups:
        if g in test_groups:
            continue
        target = int(np.argmin(fold_sizes))
        fold_of_group[g] = target
        fold_sizes[target] += int(np.sum(groups == g))

    for fold_idx in range(k):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        train_idx = [i for i in dev_indices if fold_of_group[groups[i]] != fold_idx]
        val_idx = [i for i in dev_indices if fold_of_group[groups[i]] == fold_idx]

        train_lists = {key: [file_lists[key][i] for i in train_idx] for key in file_lists}
        val_lists = {key: [file_lists[key][i] for i in val_idx] for key in file_lists}

        np.save(os.path.join(fold_dir, 'train_file_lists.npy'), train_lists)
        np.save(os.path.join(fold_dir, 'val_file_lists.npy'), val_lists)

        processor = MultiModalDataProcessor()
        stats = processor.compute_dataset_statistics(train_lists, sample_size=50, seed=seed)
        np.save(os.path.join(fold_dir, 'data_stats.npy'), stats)

        print(f"Fold {fold_idx}: Train {len(train_idx)}, Val {len(val_idx)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare IsoPlume-Net data splits")
    parser.add_argument('--data_dir', required=True,
                        help="Directory containing the co, fire, rgb1, false_rgb and mask folders")
    parser.add_argument('--output_dir', required=True,
                        help="Directory for the split file lists and statistics")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_test', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_folders = {
        key: os.path.join(args.data_dir, key)
        for key in ['co', 'fire', 'rgb1', 'false_rgb', 'mask']
    }

    file_lists = create_file_lists(data_folders)
    prepare_k_fold_data(file_lists, args.output_dir, k=args.k,
                        n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()