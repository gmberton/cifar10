
import os
import torch
import argparse
import torchvision

import models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed_weights", type=int, default=0, help="_")
parser.add_argument("--model", type=str, default="r9", choices=["r9", "r18"], help="_")

args = parser.parse_args()

if args.model == "r18":
    model = torchvision.models.resnet18(pretrained=False)
elif args.model == "r9":
    model = models.build_network()

os.makedirs(f"models_{args.model}", exist_ok=True)
torch.save(model.state_dict(), f"models_{args.model}/{args.seed_weights:02d}.pth")

