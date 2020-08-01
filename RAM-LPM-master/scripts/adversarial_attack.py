import argparse
import logging
import time

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw
from PIL import Image, ImageOps
import torch
from torchvision import transforms, datasets
from torchvision.transforms import functional as F

from tonbo.ram_lpm import RAM_LPM

torch.manual_seed(0)
FORMAT = "%(asctime)-15s %(message)s"
DEVICE = torch.device("cuda")


def attack_with_panda(attack):
    np_img = get_panda_image()
    img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(DEVICE)
    label = torch.tensor([388]).long().to(DEVICE)
    logging.info("original prediction {}".format(model(img)))
    tic = time.time()
    adv_img = attack.perturb(img, label)
    toc = time.time()
    logging.info("elapsed time: {} sec".format(toc - tic))
    logging.info(
        "prediction of purturbed image (pgd): {}".format(torch.max(model(adv_img), 1))
    )
    logging.info("adversary prediction {}".format(model(adv_img)))


def attack_with_imagenet(attack, pad_square, max_iter):
    if pad_square:
        transform = transforms.Compose([transforms.ToTensor()])

    else:
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
        )
    torch.manual_seed(0)
    val_dataset = datasets.ImageFolder("val", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    success = 0
    failure = 0
    incorrect = 0
    for i, (x, y) in enumerate(val_loader):
        if i == max_iter:
            break
        x = x.to(DEVICE)
        y = torch.tensor([y]).long().to(DEVICE)
        logging.info("y = {}".format(y.cpu().numpy()))
        torch.manual_seed(22)
        if pad_square:
            h = x.shape[2]
            w = x.shape[3]
            if h < w:
                x_padded = torch.nn.functional.pad(
                    x, (0, 0, int((w - h) / 2), int((w - h) / 2))
                )
            elif h > w:
                x_padded = torch.nn.functional.pad(
                    x, (int((h - w) / 2), int((h - w) / 2), 0, 0)
                )
            orig_prediction = model(x_padded)
        else:
            orig_prediction = model(x)
        logging.info("original prediction: {}".format(orig_prediction))
        if torch.max(orig_prediction, 1)[1].cpu().numpy()[0] == y.cpu().numpy()[0]:
            logging.info("original prediction matches the label.")
        else:
            logging.info("original prediction does not match the label.")
            incorrect += 1
            continue
        tic = time.time()
        x_adv = attack.perturb(x, y)
        toc = time.time()
        logging.info("elapsed time: {} sec".format(toc - tic))
        if pad_square:
            h = x.shape[2]
            w = x.shape[3]
            if h < w:
                x_adv_padded = torch.nn.functional.pad(
                    x_adv, (0, 0, int((w - h) / 2), int((w - h) / 2))
                )
            elif h > w:
                x_adv_padded = torch.nn.functional.pad(
                    x_adv, (int((h - w) / 2), int((h - w) / 2), 0, 0)
                )
            prediction_on_purturbed_img = model(x_adv_padded)
        else:
            prediction_on_purturbed_img = model(x_adv)

        logging.info(
            "argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1))
        )
        logging.info(
            "prediction of purturbed image: {}".format(prediction_on_purturbed_img)
        )
        if (
            torch.max(prediction_on_purturbed_img, 1)[1].cpu().numpy()[0]
            == y.cpu().numpy()[0]
        ):
            failure += 1
            logging.info("attack failed.")
        else:
            success += 1
            logging.info("attack succeeded.")
        logging.info(
            "success : failure : incorrect = {} : {} : {}".format(
                success, failure, incorrect
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", choices=["spsa", "pgd"])
    parser.add_argument("img_size", choices=["224", "original"])
    parser.add_argument("data", choices=["panda", "imagenet"])
    parser.add_argument("ckpt", help="path to checkpoint file")
    args = parser.parse_args()

    mc_sampling = 1
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
        device=DEVICE,
    )
    normalize = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    ramlpm = ramlpm.to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    ramlpm.load_state_dict(ckpt)
    model = torch.nn.Sequential(normalize, ramlpm).to(DEVICE)
    model.eval()
    if args.attack == "spsa":
        attack = LinfSPSAAttack(
            model, 2.0 / 255, nb_iter=100, nb_sample=8192, max_batch_size=64
        )
        logging.basicConfig(
            filename="./adversary_exp/spsa_sparse_{}.log".format(args.img_size),
            level=logging.DEBUG,
            format=FORMAT,
        )
        logging.info("Checkpoint used: {}".format(args.ckpt))
        if args.img_size == "224":
            pad_square = False
        else:
            pad_square = True
        if args.data == "imagenet":
            attack_with_imagenet(attack, pad_square, 300)
    else:
        logging.basicConfig(
            filename="./adversary_exp/pgd_sparse_{}.log".format(args.img_size),
            level=logging.DEBUG,
            format=FORMAT,
        )
        logging.info("Checkpoint used: {}".format(args.ckpt))
        for nb_iter in [
            300,
        ]:
            attack = LinfPGDAttack(
                model, eps=2.0 / 255, nb_iter=nb_iter, eps_iter=0.2 / 255
            )
            logging.info(
                "Benchmark model with pgd attack. nb_iter = {}".format(nb_iter)
            )
            if args.img_size == "224":
                pad_square = False
            else:
                pad_square = True
            if args.data == "imagenet":
                attack_with_imagenet(attack, pad_square, 1000)
