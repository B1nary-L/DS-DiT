import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        hr_dir,
        lr_dir,
        ref_dir,
        transform=None,
        image_size=512,
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.image_size = image_size

        self.image_names = sorted([
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        lr_names = set(os.listdir(lr_dir))
        ref_names = set(os.listdir(ref_dir))
        self.image_names = [
            name for name in self.image_names
            if name in lr_names and name in ref_names
        ]

        print(f"Found {len(self.image_names)} matching image triplets")

    def __len__(self):
        return len(self.image_names)

    def _spatial_augment(self, imgs):
        
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        def _transform(img):
            if hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if rot90:
                img = img.transpose(Image.ROTATE_90)
            return img

        return [_transform(img) for img in imgs]

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load images
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name)
        ref_path = os.path.join(self.ref_dir, img_name)

        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        # Apply the same spatial augmentation to all three images
        #hr_img, lr_img, ref_img = self._spatial_augment([hr_img, lr_img, ref_img])

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
            ref_img = self.transform(ref_img)

        return hr_img, lr_img, ref_img


def get_sr_transforms(image_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform
