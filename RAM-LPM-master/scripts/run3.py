import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm

from tonbo.ram_lpm import RAM_LPM
from tonbo.utils import square_pad, AverageMeter

device = torch.device("cuda")


def train_ramlpm(train_dataset, val_dataset, logpath):
    fmt = "%(asctime)-15s %(message)s"
    logging.basicConfig(
        filename=logpath / "training.log", level=logging.DEBUG, format=fmt
    )
    batch_size = 64
    epochs = 50
    mc_sampling = 1
    num_workers = 4
    pin_memory = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    num_glimpses = 10
    ramlpm = RAM_LPM(
        0.16,
        1000,
        [[3, 3]] * 11,
        [
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 2],
            [1, 1],
            [3, 3],
        ],
        [
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 2],
            [1, 1],
            [3, 3],
        ],
        [3, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512],
        [[3, 3]] * 3,
        [[3, 3], [1, 1], [1, 3]],
        [[3, 3], [1, 1], [1, 3]],
        [3, 32, 32, 4],
        0.02,
        1.0,
        54,
        108,
        512,
        128,
        1,
        1,
        True,
        0,
        False,
        mc_sampling,
        num_glimpses,
        device=device,
    )
    if torch.cuda.device_count() > 1:
        logging.info("Use {} GPUs".format(torch.cuda.device_count()))
        ramlpm = torch.nn.DataParallel(ramlpm)
    ramlpm = ramlpm.to(device)
    params_in_where = []
    params_in_what = []
    params_in_baseliner = []
    for name, params in ramlpm.named_parameters():
        if "where" in name or "locator" in name:
            params_in_where.append(params)
        elif "baseline" in name:
            params_in_baseliner.append(params)
        else:
            params_in_what.append(params)
    init_lr_where = 1e-6
    init_lr = 0.0001
    init_lr_baseliner = 0.0001
    optimizer = torch.optim.Adam(
        [
            {"params": params_in_where, "lr": init_lr_where},
            {"params": params_in_baseliner, "lr": init_lr_baseliner},
            {"params": params_in_what, "lr": init_lr},
        ]
    )

    best_valid_acc = 0
    for epoch in range(epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        tic = time.time()
        with tqdm(total=len(train_dataset)) as pbar:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                ramlpm.train()
                log_probas, locs, log_pis, baselines = ramlpm(x)

                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, num_glimpses - 1)

                # compute losses for differentiable modules
                # baselines = torch.mean(baselines, dim=0)
                # baselines = baselines.repeat(self.batch_size).reshape(self.batch_size,
                #                                                      self.num_glimpses)
                loss_action = F.nll_loss(log_probas, y,)
                loss_baseline = F.mse_loss(baselines, R)
                # compute reinforce loss
                # summed over timesteps and averaged across batch

                # for invalid loss value.

                # adjusted_reward = R - masked_baselines.detach()
                adjusted_reward = R - baselines.detach()

                loss_reinforce = torch.sum(-log_pis * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)
                loss = loss_action + loss_baseline + loss_reinforce
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients and update SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(batch_size)
        accs = AverageMeter()
        ramlpm.eval()
        tic = time.time()
        correct_sum = 0
        with tqdm(total=len(val_dataset)) as pbar:
            with torch.no_grad():
                for i, (x, y) in enumerate(valid_loader):
                    x, y = x.to(device), y.to(device)
                    log_probas = ramlpm(x)
                    predicted = torch.max(log_probas, 1)[1]
                    correct = (predicted == y).float()
                    correct_sum += correct.sum()
                    acc = 100 * (correct.sum() / len(y))
                    accs.update(acc.item(), x.size()[0])
                    toc = time.time()
                    pbar.set_description(
                        ("{:.1f}s - acc: {:.3f}".format((toc - tic), acc.item()))
                    )
                    pbar.update(batch_size)
        accuracy = correct_sum.cpu().numpy() / len(val_dataset)
        logging.info("accuracy: {:.3f}".format(accuracy))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": ramlpm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            logpath / "epoch{}.ckpt".format(epoch),
        )
        if accuracy > best_valid_acc:
            best_valid_acc = accuracy
            torch.save(ramlpm.state_dict(), logpath / "best_model.ckpt")


if __name__ == "__main__":
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    imagenet_size = 512
    transform_train = transforms.Compose(
        [
            square_pad,
            transforms.Resize([imagenet_size, imagenet_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_val = transforms.Compose(
        [
            square_pad,
            transforms.Resize([imagenet_size, imagenet_size]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.ImageFolder("train", transform=transform_train)
    val_dataset = datasets.ImageFolder("val", transform=transform_val)
    p = Path("./imagenet_training")
    p.mkdir(exist_ok=True)
    train_ramlpm(train_dataset, val_dataset, p)
