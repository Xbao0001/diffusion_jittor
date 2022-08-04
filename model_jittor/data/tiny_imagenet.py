import os

from jittor import transform
from jittor.dataset import ImageFolder


def get_tiny_imagenet_dataloader(
    root='/nas/datasets/tiny_imagenet_200',
    batch_size=8, 
    num_workers=2, 
):
    train_loader = ImageFolder(
        root=os.path.join(root, 'train'),
        transform=transform.ToTensor(),
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = ImageFolder(
        root=root,
        transform=transform.ToTensor(),
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_tiny_imagenet_dataloader()
    imgs, labels = next(iter(train_loader))
    print(imgs.shape, labels)