#!/usr/bin/env python
"""
experiments on rotated mnist and fmnist datasets with various
model architectures.
"""
from pathlib import Path

import lazy
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver


import tonbo.model
from tonbo.data import RotatePrepocessDataset


ex = Experiment("paperwork-2018-kiritani")


@ex.config
def main():
    workdir = "./ex1"
    data_root = "~/.mnist"
    observer = FileStorageObserver.create(str(Path(workdir).resolve() / "config"))
    ex.observers.append(observer)
    lr = 0.01
    momentum = 0.9
    optimizer = "SGD"
    arch = "C(10, 5),MP(2),R,C(20, 5),D,MP(2),R,OFC(50),R,D,FC(10),LSMAX(1)"
    rot_angle = 0
    target_num = 1
    epochs = 10
    load_path = "./ex1"
    mode = "train_evaluate"
    use_cuda = False
    batch_size = 128
    shuffle = True
    log_interval = 20
    num_workers = 5
    dataset = "mnist"
    # kernel for conv in log polar net
    kernel_sizes_conv2d = [[2, 3], [1, 3], [1, 3]]
    # kernel for pooling in log polar net
    kernel_sizes_pool = [[1, 1], [1, 1], [2, 18]]
    strides_pool = [[1, 1], [1, 1], [2, 18]]
    # input/output dims for conv in log polar net
    kernel_dims = [1, 64, 128, 256]

    # innter radius of log polar fov.
    r_min = 0.01
    # outer radius of log polar fov.
    r_max = 0.6
    # height of mapped tensor
    H = 5
    H_linear = 0
    W = 18  # width of mapped tensor

    if not dataset in ["fmnist", "mnist"]:
        raise ValueError("choose ['fmnist', 'mnist'] as dataset")

    if not mode in ["train_evaluate", "train", "evaluate"]:
        raise ValueError("choose ['train_evaluate', 'train', 'evaluate'] as mode")

    del observer


@ex.capture
def download(_log, data_root, rot_angle, dataset, arch):
    _log.info("data_root = %s", data_root)
    if arch == "resnet18":
        transformations = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ]
    transformations_rot = [
        transforms.RandomRotation([rot_angle, rot_angle])
    ] + transformations
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(transformations),
        )
        test_dataset = datasets.MNIST(
            data_root, train=False, transform=transforms.Compose(transformations_rot)
        )
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(transformations),
        )
        test_dataset = datasets.FashionMNIST(
            data_root, train=False, transform=transforms.Compose(transformations_rot)
        )
    _log.info(f"set dataset as {dataset}")

    return train_dataset, test_dataset


@ex.capture
def get_iterator(_log, rot_angle, data_root, batch_size, shuffle, num_workers):
    _log.info("rot_angle = %s", rot_angle)
    train_dataset, test_dataset = download()
    train_iterator = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_iterator = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_iterator, test_iterator


@ex.capture
def train(
    _log,
    epochs,
    use_cuda,
    arch,
    lr,
    momentum,
    log_interval,
    kernel_sizes_conv2d,
    kernel_sizes_pool,
    kernel_dims,
    r_min,
    r_max,
    H,
    W,
    strides_pool,
):
    _log.info("arch = %s", arch)
    _log.info("lr = %s", lr)
    _log.info("momentum = %s", momentum)
    _log.info("use_cuda = %s", use_cuda)
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")
    if arch == "CNN2FC2":
        model = tonbo.model.CNN2FC2().to(device)
    elif arch == "CNN2FC2V2":
        model = tonbo.model.CNN2FC2V2().to(device)
    elif arch == "Net":
        model = tonbo.model.Net().to(device)
    elif arch == "logpolar":
        lksc = len(kernel_sizes_conv2d)
        lksp = len(kernel_sizes_pool)
        lsp = len(strides_pool)
        lkd = len(kernel_dims)
        if not lksc == lksp == lsp == (lkd - 1):
            raise ValueError("Check the kernel sizes, strides size")
        model = tonbo.model.LogPolarConv(
            kernel_sizes_conv2d,
            kernel_sizes_pool,
            strides_pool,
            kernel_dims,
            r_min,
            r_max,
            H,
            W,
            upsampling_factor_r=10,
            upsampling_factor_theta=10,
        ).to(device)
    if arch == "resnet18":
        model = tonbo.model.Res18()
    else:
        model = lazy.LazyNet(arch).to(device)
        model.dryrun(torch.randn(1, 1, 28, 28))

    _log.info(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_iterator, test_iterator = get_iterator()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_iterator):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # summary.histgram(model, batch_idx)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_iterator.dataset),
                        100.0 * batch_idx / len(train_iterator),
                        loss.item(),
                    )
                )

        model.eval()
        evaluate(epoch=epoch, model=model, test_iterator=test_iterator)
        model.train()


@ex.capture
def evaluate(_run, _log, use_cuda, epoch, model, test_iterator):
    test_loss = 0
    correct = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for data, target in test_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, size_average=False
            ).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_iterator.dataset)
    _run.log_scalar("test.acc", 100.0 * correct / len(test_iterator.dataset), 0)
    _run.log_scalar("test.loss", test_loss, epoch)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_iterator.dataset),
            100.0 * correct / len(test_iterator.dataset),
        )
    )


@ex.automain
def main(_log, epochs):
    _log.info("hello")
    train()
