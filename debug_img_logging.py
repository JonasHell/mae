import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def main():
    img01 = torch.rand(3, 224, 224)
    img255 = img01 * 255
    print(img01.size(), img255.size())
    img01_pil = torchvision.transforms.functional.to_pil_image(img01)
    img255_pil = torchvision.transforms.functional.to_pil_image(img255)
    print(img01_pil.size, img255_pil.size)

    logger = SummaryWriter(log_dir="output_dir")
    logger.add_image("img01", img01, dataformats="CHW")
    logger.add_image("img255", img255, dataformats="CHW")
    # can't log PIL images
    # logger.add_image("img01_pil", img01_pil, dataformats="HWC")
    # logger.add_image("img255_pil", img255_pil, dataformats="HWC")
    logger.close()


if __name__ == "__main__":
    main()
