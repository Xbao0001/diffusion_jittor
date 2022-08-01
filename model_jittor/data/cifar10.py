from jittor import transform
from jittor.dataset import CIFAR10, CIFAR100


def get_cifar10_dataloader( 
    batch_size=8, 
    num_workers=2, 
):
    train_transform = transform.Compose([ 
        transform.CenterCrop(),
        transform.ToTensor(),
    ])
    val_transform = transform.Compose([
        transform.CenterCrop(),
        transform.ToTensor(),
    ])

    train_loader = CIFAR10(
        root='/nas/datasets/cifar10_jittor',
        train=True,
        transform=train_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = CIFAR10(
        root='/nas/datasets/cifar10_jittor',
        train=False,
        transform=val_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True,
    )
    return train_loader, val_loader