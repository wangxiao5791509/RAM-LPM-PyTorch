import argparse
import glob
import logging
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from tqdm import tqdm

from tonbo.ram_lpm import RAM_LPM
from tonbo.utils import square_pad, AverageMeter
from tonbo.modules import action_network


def run(soft_attention, log_path, ckpt_path):
    torch.manual_seed(0)
    DEVICE = torch.device("cuda")
    logging.basicConfig(
        filename=log_path, level=logging.DEBUG, format="%(asctime)-15s %(message)s"
    )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    imagenet_size = 512
    transform_val = transforms.Compose(
        [
            square_pad,
            transforms.Resize([imagenet_size, imagenet_size]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder("val", transform=transform_val)
    class_num = 1000
    batch_size = 64
    num_workers = 4
    pin_memory = True
    accuracies = []

    for num_glimpses in range(1, 15):
        for mc_sampling in [1, 5, 10, 20]:
            valid_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            ramlpm = RAM_LPM(
                0.16,
                class_num,
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
                device=DEVICE,
                soft_attention=soft_attention
            )
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            ramlpm.load_state_dict(ckpt)
            ramlpm.eval()
            ramlpm = ramlpm.to(DEVICE)
            tic = time.time()
            correct_sum = 0
            with tqdm(total=len(val_dataset)) as pbar:
                with torch.no_grad():
                    for i, (x, y) in enumerate(valid_loader):
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        log_probas = ramlpm(x)
                        predicted = torch.max(log_probas, 1)[1]
                        correct = (predicted == y).float()
                        correct_sum += correct.sum()
                        acc = 100 * (correct.sum() / len(y))
                        toc = time.time()
                        pbar.set_description(
                            (
                                "{:.1f}s - acc: {:.3f}".format(
                                    (toc - tic), acc.item()
                                )
                            )
                        )
                        pbar.update(batch_size)
            accuracy = correct_sum.cpu().numpy() / len(val_dataset)
            logging.info(
                "mc_sample: {}, num_glimpse: {}, accuracy: {}".format(
                    mc_sampling, num_glimpses, accuracy
                )
            )
            accuracies.append(accuracy)

    accuracies = np.array(accuracies)
    accuracies = accuracies.reshape([-1, 2])
    accuracies = np.fliplr(accuracies)
    
    fig = plt.figure()
    plt.plot(range(1, accuracies.shape[0] + 1), accuracies)
    plt.title("accuracy")
    plt.xlabel("number of glimpse")
    plt.ylabel("accuracy")
    plt.legend(["MC sampling = 20","MC sampling = 10","MC sampling = 5", "MC sampling = 1"])
    fig.savefig(os.path.join(os.path.dirname(log_path), "summary.png"), dpi=500)
    np.save(os.path.join(os.path.dirname(log_path), "summary.npy"), accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_attention", choices=["none", "se", "se_softmax", "cbam", "cbam_softmax"])
    parser.add_argument("log_path", help="path to log file") 
    parser.add_argument("ckpt", help="path to checkpoint")
    args = parser.parse_args()
    run(args.soft_attention, args.log_path, args.ckpt)
