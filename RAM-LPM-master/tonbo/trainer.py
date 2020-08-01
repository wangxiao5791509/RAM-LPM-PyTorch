import os
import time
import shutil
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
from .utils import AverageMeter
from .model import RecurrentAttention


def get_logger(logname, loglevel="DEBUG"):
    _logger = getLogger(logname)
    handler = StreamHandler()
    handler.setLevel(loglevel)
    fmt = Formatter("%(asctime)s %(levelname)6s %(funcName)20s : %(message)s")
    handler.setFormatter(fmt)
    _logger.setLevel(loglevel)
    _logger.addHandler(handler)
    _logger.propagate = False
    return _logger


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(
        self,
        model,
        model_name,
        optimizer,
        train_loader,
        valid_loader,
        test_loader,
        is_train: bool = True,
        mc_sampling: int = 10,
        batch_size: int = 32,
        num_glimpses: int = 6,
        epochs: int = 200,
        use_gpu: bool = True,
        best: bool = False,
        work_dir: str = "ex1",
        resume: bool = False,
        *,
        logger=None,
        sacred_run=None,
        std_half_time=None,
    ):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data loader
        """
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.is_train = is_train
        if is_train:
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
            self.num_test = len(self.test_loader.dataset)
        else:
            self.num_test = len(self.test_loader.dataset)

        # reinforce params
        self.mc_sampling = mc_sampling
        self.num_glimpses = num_glimpses

        # training params
        self.epochs = epochs
        self.start_epoch = 0
        self.batch_size = batch_size

        # misc params
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.best = best
        self.work_dir = str(Path(work_dir).expanduser().resolve())
        self.best_valid_acc = 0.0
        self.counter = 0
        self.resume = resume
        self.print_freq = 10
        self.plot_freq = 1

        self.plot_num = 9
        self.plot_num = (
            self.plot_num if self.batch_size > self.plot_num else self.batch_size
        )

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

        self.sacred_run = sacred_run
        self.writer = SummaryWriter(log_dir=(self.work_dir + "/logs"))
        self.std_half_time = std_half_time

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        self.logger.info(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            # train for 1 epoch
            if self.model_name in ["rirmva", "ramlpm"]:
                self.model.train()
                train_loss, train_acc = self.train_one_epoch_rirmva(epoch)
                # evaluate on validation set
                with torch.no_grad():
                    self.model.eval()
                    valid_loss, valid_acc = self.validate_rirmva(epoch)
                if self.std_half_time:
                    decreased_std = self.model.std * (
                        (0.5) ** (1.0 / self.std_half_time)
                    )
                    print("std decreased")
                    print(decreased_std)
                    self.model.std = decreased_std
                    self.model.locator.std = decreased_std
            else:
                train_loss, train_acc = self.train_one_epoch(epoch)
                # evaluate on validation set
                with torch.no_grad():
                    valid_loss, valid_acc = self.train_one_epoch(epoch, mode="valid")

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            self.logger.info(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            self.sacred_run.log_scalar("train.loss", train_loss, epoch)
            self.sacred_run.log_scalar("train.acc", train_acc, epoch)
            self.sacred_run.log_scalar("valid.loss", valid_loss, epoch)
            self.sacred_run.log_scalar("valid.acc", valid_acc, epoch)
            # check for improvement
            if not is_best:
                self.counter += 1

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def one_step_rmva(self, istep, x, y):
        # initialize location vector and hidden state
        return loss, acc

    def train_one_epoch(self, epoch=0, mode="train"):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        if mode == "train":
            loader = self.train_loader
            total = self.num_train
            self.model = self.model.train()
        elif mode == "valid":
            loader = self.valid_loader
            total = self.num_valid
            self.model = self.model.eval()
        elif mode == "test":
            loader = self.test_loader
            total = self.num_test
            self.model = self.model.eval()

        with tqdm(total=total) as pbar:
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                pred = F.log_softmax(pred, dim=-1)
                loss = F.nll_loss(pred, y)
                pred = pred.argmax(dim=1, keepdim=True)
                correct = pred.eq(y.view_as(pred)).sum().item()
                acc = 100 * (correct / len(y))
                losses.update(loss.item(), x.size()[0])
                accs.update(acc, x.size()[0])
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:03d} {:6s}:{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            epoch, mode, (toc - tic), loss.item(), acc
                        )
                    )
                )
                self.writer.add_scalar(f"data/{mode}-loss", loss.item(), i)
                self.writer.add_scalar(f"data/{mode}-acc", acc, i)

                pbar.update(self.batch_size)

        return losses.avg, accs.avg

    def train_one_epoch_rirmva(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True
                self.batch_size = x.shape[0]
                W = x.shape[2]
                H = x.shape[2]

                h_t, l_t = self.model.reset(self.batch_size, self.device)

                # save images
                imgs = []
                imgs.append(x[0 : self.plot_num])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []

                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # store
                    locs.append(l_t[0 : self.plot_num])
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # log_pi.append(p)
                # baselines.append(b_t)
                locs.append(l_t[0 : self.plot_num])

                # convert list to tensors and reshape

                baselines = torch.stack(baselines, dim=1)
                log_pi = torch.stack(log_pi, dim=1)

                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses - 1)

                # compute losses for differentiable modules
                # baselines = torch.mean(baselines, dim=0)
                # baselines = baselines.repeat(self.batch_size).reshape(self.batch_size,
                #                                                      self.num_glimpses)
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)
                # compute reinforce loss
                # summed over timesteps and averaged across batch

                # for invalid loss value.

                # adjusted_reward = R - masked_baselines.detach()
                adjusted_reward = R - baselines.detach()

                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)
                loss = loss_action + loss_baseline + loss_reinforce
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time. only update every 100 steps. this part
                # takes ~40 ms.
                if i % 100 == 0:
                    # updating pbar takes ~40 ms.
                    toc = time.time()
                    batch_time.update(toc - tic)

                    pbar.set_description(
                        (
                            "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                                (toc - tic), loss.item(), acc.item()
                            )
                        )
                    )
                    pbar.update(self.batch_size * 100)

                    # this logging takes ~40 ms as well
                    self.writer.add_scalar(
                        "data/train-loss-action", loss_action.item(), i
                    )
                    self.writer.add_scalar(
                        "data/train-loss-reinforce", loss_reinforce.item(), i
                    )
                    self.writer.add_scalar(
                        "data/train-loss-baseline", loss_baseline.item(), i
                    )
                    self.writer.add_scalar("data/train-loss", loss.item(), i)
                    self.writer.add_scalar("data/train-acc", acc.item(), i)

        return losses.avg, accs.avg

    def _rescale(self, x, _min, _max):
        x += 1
        return int(x * (_max - _min) / 2)

    def validate_rirmva(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate 10 times
            x = x.repeat(self.mc_sampling, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.model.reset(self.batch_size, self.device)

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(self.mc_sampling, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(
                self.mc_sampling, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.mc_sampling, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss

            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            self.writer.add_scalar("data/valid-loss-action", loss_action.item(), i)
            self.writer.add_scalar(
                "data/valid-loss-reinforce", loss_reinforce.item(), i
            )
            self.writer.add_scalar("data/valid-loss-baseline", loss_baseline.item(), i)
            self.writer.add_scalar("data/valid-loss", loss.item(), i)
            self.writer.add_scalar("data/valid-acc", acc.item(), i)
            #
            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])
            # log to tensorboard

        return losses.avg, accs.avg

    def test(self):
        if self.model_name in ["rirmva", "ramlpm"]:
            msg = self.test_rirmva()
        else:
            loss, acc = self.train_one_epoch(mode="test")
            msg = f"[*] Test Acc: {acc:.2f}%"
            self.logger.info(msg)
        return msg

    def test_rirmva(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)

                # duplicate 10 times
                x = x.repeat(self.mc_sampling, 1, 1, 1)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.model.reset(self.batch_size, self.device)

                # extract the glimpses
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # last iteration
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

                log_probas = log_probas.view(self.mc_sampling, -1, log_probas.shape[-1])
                log_probas = torch.mean(log_probas, dim=0)

                pred = log_probas.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        msg = "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
            correct, self.num_test, perc, error
        )
        self.logger.info(msg)
        return msg

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """

        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = Path(self.work_dir) / filename
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, Path(self.work_dir) / filename)

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        self.logger.info("[*] Loading model from {}".format(self.work_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = Path(self.work_dir) / filename
        ckpt = torch.load(str(ckpt_path))

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            self.logger.info(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            self.logger.info(
                "[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"])
            )
