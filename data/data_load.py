import os
from PIL import Image as Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor


def _is_image_file(name: str) -> bool:
    ext = name.split('.')[-1].lower()
    return ext in ['png', 'jpg', 'jpeg']


def _has_flat_split(root: str) -> bool:
    return os.path.isdir(os.path.join(root, 'blur')) and os.path.isdir(os.path.join(root, 'sharp'))


def _has_hierarchical_split(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    for scene in os.listdir(root):
        scene_path = os.path.join(root, scene)
        if not os.path.isdir(scene_path):
            continue
        if _has_flat_split(scene_path):
            return True
    return False


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True, proportion: float = 1.0, crop_size: int = 256):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transforms_list = []
        if crop_size and crop_size > 0:
            transforms_list.append(PairRandomCrop(crop_size))
        transforms_list.extend([PairRandomHorizontalFilp(), PairToTensor()])
        transform = PairCompose(transforms_list)

    if _has_flat_split(image_dir):
        base_dataset = DeblurDataset(image_dir, transform=transform)
    else:
        base_dataset = HierarchicalDeblurDataset(image_dir, transform=transform)

    # Apply proportion subset if needed
    if proportion is None:
        proportion = 1.0
    proportion = max(0.0, min(1.0, float(proportion)))
    if proportion <= 0:
        raise ValueError('train_dataloader proportion must be > 0')
    if proportion < 1.0:
        total = len(base_dataset)
        count = max(1, int(total * proportion))
        idxs = list(range(count))
        dataset = Subset(base_dataset, idxs)
    else:
        dataset = base_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def _choose_eval_split(path: str):
    candidates = [os.path.join(path, 'test'), os.path.join(path, 'valid'), path]
    for c in candidates:
        if _has_flat_split(c):
            return c, True
        if _has_hierarchical_split(c):
            return c, False
    raise FileNotFoundError(f"No split with blur/sharp found under {path}. Tried: {candidates}")


def test_dataloader(path, batch_size=1, num_workers=0):
    split_root, is_flat = _choose_eval_split(path)
    dataset = DeblurDataset(split_root, is_test=True) if is_flat else HierarchicalDeblurDataset(split_root, is_test=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def valid_dataloader(path, batch_size=1, num_workers=0):
    split_root, is_flat = _choose_eval_split(path)
    dataset = DeblurDataset(split_root) if is_flat else HierarchicalDeblurDataset(split_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test: bool = False):
        self.image_dir = image_dir
        blur_dir = os.path.join(image_dir, 'blur')
        if not os.path.isdir(blur_dir):
            raise FileNotFoundError(f"Blur dir not found: {blur_dir}")
        self.image_list = [f for f in os.listdir(blur_dir) if _is_image_file(f)]
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blur_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, 'blur', blur_name))
        label = Image.open(os.path.join(self.image_dir, 'sharp', blur_name))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            return image, label, blur_name
        return image, label


class HierarchicalDeblurDataset(Dataset):
    """
    Dataset that supports multiple scene subfolders under a split.
    Expected layout:
      <split_root>/<scene>/{blur,sharp}/<filename>
    """

    def __init__(self, split_root: str, transform=None, is_test: bool = False):
        self.transform = transform
        self.is_test = is_test
        self.pairs = []  # list of (blur_path, sharp_path)
        self.names = []  # relative names for saving (scene/filename)

        if not os.path.isdir(split_root):
            raise ValueError(f'Split root not found: {split_root}')

        for scene in sorted(os.listdir(split_root)):
            scene_path = os.path.join(split_root, scene)
            if not os.path.isdir(scene_path):
                continue
            blur_dir = os.path.join(scene_path, 'blur')
            sharp_dir = os.path.join(scene_path, 'sharp')
            if not (os.path.isdir(blur_dir) and os.path.isdir(sharp_dir)):
                continue

            img_list = [f for f in os.listdir(blur_dir) if _is_image_file(f)]
            img_list.sort()
            for name in img_list:
                blur_path = os.path.join(blur_dir, name)
                sharp_path = os.path.join(sharp_dir, name)
                if os.path.isfile(blur_path) and os.path.isfile(sharp_path):
                    self.pairs.append((blur_path, sharp_path))
                    self.names.append(os.path.join(scene, name))

        if len(self.pairs) == 0:
            raise ValueError(f"No images found under {split_root}. Expected structure: <split>/<scene>/{{blur,sharp}}/")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        image = Image.open(blur_path)
        label = Image.open(sharp_path)

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.names[idx] if idx < len(self.names) else os.path.basename(blur_path)
            return image, label, name
        return image, label
