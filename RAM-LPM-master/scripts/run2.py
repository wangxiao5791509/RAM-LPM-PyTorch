# /uer/bin/env python
"""
Experiment code for recurrent visual attention model
with a circular field of view.
"""
from pathlib import Path
import os

import gdown
import numpy as np
from PIL import ImageFile
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchex.data import transforms as extransforms
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_tensor

import tonbo.model
import tonbo.trainer
from tonbo.data import RotatePrepocessDataset, ExpandDataset
from tonbo.utils import square_pad
from polar_transformer_networks.util import train_test_val_mnist

ImageFile.LOAD_TRUNCATED_IMAGES = True
ex = Experiment("recurrent-visual-attention")


class Pass(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img


@ex.config
def config():
    """
    Defines configuration used in a sacred experiment.
    """
    work_dir = "./rva-exp01"  # save resuls of experiment in this directory
    observer = FileStorageObserver.create(str(Path(work_dir).resolve() / "config"))
    ex.observers.append(observer)

    # glimpse network params
    patch_size = (
        8  # size of extracted patch at highest res. used if model_name is "rirmva".
    )
    glimpse_scale = 2  # scale of successive patches, used if model_name is "rirmva".
    num_patches = (
        1  # of downscaled patches per glimpse. used if model_name is "rirmva".
    )
    sampling_interval = (
        1 / 28
    )  # sampling interval in the smallest grid for rectangular fov. used if model_name is "rirmva".
    h_g = 128  # hidden layer size of the fc layer for `phi`. used only in rirmva.
    h_l = 128  # hidden layer size of the fc layer for `l`. used only in rirmva.
    hidden_size = 256  # hidden size of rnn. used only in rirmva.

    # determine how finely pixel values are sampled in log-polar coords to compute the integral along the r axis. used only in BrainNet.
    upsampling_factor_r = 10
    upsampling_factor_theta = 10  # determine how finely pixel values are sampled in log-polar coords to compute the integral along the angular axis. used only in BrainNet.
    hidden_what = 128  # node num for recurrent what pathway. used only in BrainNet.
    hidden_where = 128  # node num for recurrent where pathway. used only in BrainNet.
    # kernel sizes for convolution in glimpse network. used only in BrainNet.
    kernel_sizes_conv2d = [[1, 3], [1, 3], [2, 3]]
    # kernel sizes for max pooling in glimpse network. used only in BrainNet.
    kernel_sizes_pool = [[1, 1], [1, 1], [2, 12]]
    strides_pool = kernel_sizes_pool
    kernel_dims = [
        1,
        64,
        128,
        256,
    ]  # output dims for convolution in what pathway. used only in BrainNet.
    # kernel sizes for convolution in glimpse network. used only in BrainNet.
    kernel_sizes_conv2d_where = [[1, 3], [1, 3], [2, 3]]
    # kernel sizes for max pooling in glimpse network. used only in BrainNet.
    kernel_sizes_pool_where = [[3, 3], [1, 1], [2, 3]]
    strides_pool_where = kernel_sizes_pool_where
    kernel_dims_where = [
        1,
        32,
        32,
        4,
    ]  # output dims for convolution in what pathway. used only in BrainNet.
    r_min = 0.01  # r_min for log-polar sampling space. used only in BrainNet.
    r_max = 0.6  # r_max for log-polar sampling space. used only in BrainNet.
    H = 5  # height for input tensor. used only in BrainNet.
    H_linear = 0
    W = 12  # width for input tensor. used only in BrainNet.
    log_r = True  # switch polar vs log polar sampling
    use_resnet_in_br = False

    num_glimpses = 6  # # of glimpses, i.e. BPTT iterations
    model_name = (
        "rirmva"  # Rotation Invariance Recurrent Models of Visual Attention (our model)
    )

    # reinfoce params
    std = 0.17  # gaussian policy standard deviation
    mc_sampling = 10  # Monte Carlo sampling for valid and test sets

    # data params
    valid_size = 0.1  # percentage split of the training set used for the validation set. Should be a float in the range [0, 1]. In the paper, this number is set to 0.1.
    batch_size = 32  # batch size
    num_workers = 4  # number of subprocesses to use when loading the dataset.
    shuffle: bool = True  # whether to shuffle the train/validation indices.
    dataset: str = "mnist"  # you can choose dataset type from [mnist, fmnist, imagenet]
    data_root = "~/." + dataset  # save raw data in this directory
    imagenet_size: int = 512
    pin_memory: bool = False  # whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    num_classes: int = 10  # the number of class in a dataset.
    num_channels: int = 1  # channel of input image.
    rot_angle: float = 0  # rotate test images with this rotation angle.
    expand: bool = False  # you can increase the size of an image
    width: int = 60  # width is increased to this value.
    height: int = 60  # height is increased to this value.
    background: float = 0.0  # if expand is True, this value is used as a default value of padding.

    # training params
    is_train = True  #
    momentum = 0.9  # for SDG
    epochs = 200  #
    init_lr = 1e-3  #
    init_lr_where = 1e-6
    init_lr_baseliner = 1e-4
    optimizer_type = "Adam"  # optimizer type

    # other params
    use_gpu = True  #
    best = True  # use the model has best accuracy in validation data when training.
    resume = False  # reload the trained model from work_dir
    benchmark = True  # use torch.cuda.benchmark mode to speed up traning.

    rotate_traindata = True  # rotate train data
    rotate_testdata = True  # rotate test data
    std_half_time = None  # determines how fast std decays. half time in epochs.

    del observer


@ex.capture
def download(
    _log,
    data_root,
    rot_angle,
    dataset,
    expand,
    width,
    height,
    background,
    rotate_traindata,
    rotate_testdata,
    imagenet_size,
):
    _log.info("data_root = %s", data_root)

    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        test_dataset = datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
    elif dataset == "rotatedmnist":
        trainX, trainY, valX, valY, testX, testY = train_test_val_mnist(
            data_root, one_hot=False
        )
        trainX = np.expand_dims(np.squeeze(trainX), 1)
        valX = np.expand_dims(np.squeeze(valX), 1)
        testX = np.expand_dims(np.squeeze(testX), 1)
        trainY = np.squeeze(trainY.astype(np.int64))
        valY = np.squeeze(valY.astype(np.int64))
        testY = np.squeeze(testY.astype(np.int64))
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(trainX), torch.from_numpy(trainY)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(valX), torch.from_numpy(valY)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(testX), torch.from_numpy(testY)
        )
    elif dataset == "mnistr":
        """
        half-rotated mnist
        """
        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.RandomRotation((-90, 90)), transforms.ToTensor(),]
            ),
        )
        test_dataset = datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [transforms.RandomRotation((-90, 90)), transforms.ToTensor(),]
            ),
        )
    elif dataset == "mnistrts":
        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    extransforms.RandomResize(19, 34, True),
                    transforms.RandomRotation((-45, 45)),
                    transforms.PadRandomSift(42, 42),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [
                    extransforms.RandomResize(19, 34, True),
                    transforms.RandomRotation((-45, 45)),
                    transforms.PadRandomSift(42, 42),
                    transforms.ToTensor(),
                ]
            ),
        )
        assert expand == False, "mnistrts not support exapnd = True option"
    elif dataset == "sim2mnist":
        Path(data_root).mkdir(parents=True, exist_ok=True)
        if not (os.path.isfile(os.path.join(data_root, "sim2mnist.npz"))):
            gdown.download(
                "https://drive.google.com/uc?id=1AtVj4xqdshyYEOMU7EwvrVjcZkgBDZCL",
                os.path.join(data_root, "sim2mnist.npz"),
                quiet=False,
            )
        npzfile = np.load(os.path.join(data_root, "sim2mnist.npz"))
        x_train = torch.from_numpy(
            np.expand_dims(npzfile["x_train"].astype(np.float32), axis=1)
        )
        x_val = torch.from_numpy(
            np.expand_dims(npzfile["x_validation"].astype(np.float32), axis=1)
        )
        x_test = torch.from_numpy(
            np.expand_dims(npzfile["x_test"].astype(np.float32), axis=1)
        )
        y_train = torch.from_numpy(np.argmax(npzfile["y_train"], axis=1))
        y_val = torch.from_numpy(np.argmax(npzfile["y_validation"], axis=1))
        y_test = torch.from_numpy(np.argmax(npzfile["y_test"], axis=1))
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        assert expand == False, "sim2mnist not support exapnd = True option"

    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        test_dataset = datasets.FashionMNIST(
            data_root,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
    elif dataset == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = datasets.ImageFolder("train", transform=transform_train)
        test_dataset = datasets.ImageFolder("val", transform=transform_test)
    elif dataset == "imagenet_full":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_train = transforms.Compose(
            [
                square_pad,
                transforms.Resize([imagenet_size, imagenet_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                square_pad,
                transforms.Resize([imagenet_size, imagenet_size]),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = datasets.ImageFolder("train", transform=transform_train)
        test_dataset = datasets.ImageFolder("val", transform=transform_test)
    elif dataset == "imagenet-a":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize([imagenet_size, imagenet_size]),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_dataset = datasets.ImageFolder(
            "imagenet-a-processed", transform=transform_test
        )
        return test_dataset

    train_dataset = RotatePrepocessDataset(train_dataset)
    test_dataset = RotatePrepocessDataset(test_dataset, rotation_angle=rot_angle)
    if expand:
        train_dataset = ExpandDataset(train_dataset, width, height, background)
        test_dataset = ExpandDataset(test_dataset, width, height, background)

    _log.info(f"set dataset as {dataset}")
    if dataset in ["rotatedmnist", "sim2mnist"]:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset


@ex.capture
def get_loaders(
    _seed,
    _log,
    rot_angle,
    batch_size,
    dataset,
    shuffle,
    num_workers,
    valid_size,
    pin_memory,
    mc_sampling,
):
    _log.info("rot_angle = %s", rot_angle)
    if dataset in ["sim2mnist", "rotatedmnist"]:
        train_dataset, val_dataset, test_dataset = download()
    elif dataset == "imagenet-a":
        test_dataset = download()
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size // mc_sampling,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return None, None, test_loader
    else:
        train_dataset, test_dataset = download()
        num_train = len(train_dataset)
        val_size = int(np.floor(valid_size * num_train))
        train_size = num_train - val_size
        # seed this part
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size // mc_sampling,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size // mc_sampling,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


@ex.capture
def get_optimizer(
    model, optimizer_type, init_lr, init_lr_where, init_lr_baseliner, momentum
):
    # TODO: how to feed momentum, regardless optimizer type.
    params_in_where = []
    params_in_what = []
    params_in_baseliner = []
    for name, params in model.named_parameters():
        if "where" in name or "locator" in name:
            params_in_where.append(params)
        elif "baseline" in name:
            params_in_baseliner.append(params)
        else:
            params_in_what.append(params)
    return getattr(optim, optimizer_type)(
        [
            {"params": params_in_where, "lr": init_lr_where},
            {"params": params_in_baseliner, "lr": init_lr_baseliner},
            {"params": params_in_what, "lr": init_lr},
        ]
    )


@ex.capture
def get_model(
    model_name,
    patch_size,
    num_patches,
    glimpse_scale,
    num_channels,
    h_g,
    h_l,
    std,
    hidden_size,
    num_classes,
    kernel_sizes_conv2d,
    kernel_sizes_pool,
    strides_pool,
    kernel_dims,
    kernel_sizes_conv2d_where,
    kernel_sizes_pool_where,
    strides_pool_where,
    kernel_dims_where,
    r_min,
    r_max,
    H,
    W,
    sampling_interval,
    hidden_what,
    hidden_where,
    upsampling_factor_r,
    upsampling_factor_theta,
    log_r,
    H_linear,
    use_resnet_in_br,
):
    model = None
    import tonbo.model

    if model_name == "rirmva":
        model = tonbo.model.RecurrentAttention(
            patch_size,
            num_patches,
            glimpse_scale,
            num_channels,
            h_g,
            h_l,
            std,
            hidden_size,
            num_classes,
            sampling_interval,
        )
    elif model_name == "resnet18":
        import tonbo.resnet

        model = tonbo.resnet.resnet18(num_classes=10)
    elif model_name == "resnet34":
        import tonbo.resnet

        model = tonbo.resnet.resnet34(num_classes=10)
    elif model_name == "resnet50":
        import tonbo.resnet

        model = tonbo.resnet.resnet50(num_classes=10)
    elif model_name == "simplecnn":
        model = tonbo.model.Net()
    elif model_name == "ramlpm":
        model = tonbo.model.RAM_LPM(
            std,
            num_classes,
            kernel_sizes_conv2d,
            kernel_sizes_pool,
            strides_pool,
            kernel_dims,
            kernel_sizes_conv2d_where,
            kernel_sizes_pool_where,
            strides_pool_where,
            kernel_dims_where,
            r_min,
            r_max,
            H,
            W,
            hidden_what,
            hidden_where,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            H_linear,
            use_resnet_in_br,
        )
    else:
        _models = ["rirmva", "resnet18", "resnet34", "resnet50", "simplecnn", "ramlpm"]
        raise NotImplementedError(f"model_name is only supported in {_models}")

    return model


@ex.capture
def get_trainer(
    _run,
    _log,
    model_name,
    num_glimpses,
    epochs,
    mc_sampling,
    batch_size,
    use_gpu,
    best,
    work_dir,
    is_train,
    resume,
    std_half_time,
    use_resnet_in_br,
):

    if use_gpu:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_gpu else "cpu")
    model = get_model()
    model = model.to(device)

    optimizer = get_optimizer(model=model)

    train_loader, valid_loader, test_loader = get_loaders()

    trainer = tonbo.trainer.Trainer(
        model,
        model_name,
        optimizer,
        train_loader,
        valid_loader,
        test_loader,
        is_train,
        mc_sampling,
        batch_size,
        num_glimpses,
        epochs,
        use_gpu,
        best,
        work_dir,
        resume,
        logger=_log,
        sacred_run=_run,
        std_half_time=std_half_time,
    )
    return trainer


@ex.automain
def main(_run, _log, _seed, use_gpu, is_train):
    """Runs experiments.
    Args:
       _run: live information store object. You can store some metrics, for example accuracy and loss.
       _log:
       _seed: seed used for numpy and torch.
       use_gpu: boolean for use of gpus.
       is_train: switches trainig and inference.
    """
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(_seed)
    trainer = get_trainer()
    if is_train:
        trainer.train()
        msg = trainer.test()
    else:
        msg = trainer.test()
    p = Path("~/.teamswebhook").expanduser().resolve()
    if p.exists():
        _log.info("send a message via teams web hook.")
        import pymsteams

        webhook_url = p.open().read().strip()
        myTeamsMessage = pymsteams.connectorcard(webhook_url)
        myTeamsMessage.text(f"{work_dir}\n{msg}")
        myTeamsMessage.send()
    trainer.writer.close()
