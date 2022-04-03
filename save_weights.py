
import os
import torch
import argparse
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed_weights", type=int, default=0, help="_")

args = parser.parse_args()

model = torchvision.models.resnet18(pretrained=False)

os.makedirs("models", exist_ok=True)
torch.save(model, f"models/{args.seed_weights:02d}.pth")

