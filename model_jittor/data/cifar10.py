from jittor import transform
from jittor.dataset import CIFAR10, CIFAR100


def get_cifar10_dataloader( 
    batch_size=8, 
    num_workers=2, 
):
    train_loader = CIFAR10(
        root='/nas/datasets/cifar10',
        train=True,
        transform=transform.ToTensor(),
        download=True, 
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = CIFAR10(
        root='/nas/datasets/cifar10',
        train=False,
        transform=transform.ToTensor(),
        download=True, 
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True,
    )
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_cifar10_dataloader(batch_size=16, num_workers=2)
    print(len(train_loader), len(val_loader)) # 50000 10000 which do not take acount batch size