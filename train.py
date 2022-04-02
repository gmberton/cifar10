
import sys
import torch
import logging
import argparse
import torchvision
import torchmetrics
from datetime import datetime
import torchvision.transforms as T
from torch.utils.data import Subset

import commons

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backbone", type=str, default="alexnet", choices=["alexnet", "resnet18"], help="_")
parser.add_argument("--lr", type=float, default=0.001, help="_")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
parser.add_argument("--batch_size", type=int, default=16, help="_")
parser.add_argument("--num_workers", type=int, default=3, help="_")
parser.add_argument("--epochs_num", type=int, default=100, help="_")
parser.add_argument("--seed_weights", type=int, default=0, help="_")
parser.add_argument("--seed_optimization", type=int, default=0, help="_")
parser.add_argument("--save_dir", type=str, default="default", help="_")

args = parser.parse_args()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed_optimization)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### DATASETS & DATALOADERS
OUT_SIZE = 64 if args.backbone == "alexnet" else 32
transform = T.Compose([
        T.ToTensor(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(OUT_SIZE, scale=(0.8, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# torchvision.datasets.ImageFolder()
train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_set = Subset(train_val_set, range(40000))
val_set = Subset(train_val_set, range(40000, 50000))
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.num_workers,
                                           pin_memory=(args.device == "cuda"))
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=(args.device == "cuda"))
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers,
                                          pin_memory=(args.device == "cuda"))

#### MODEL & CRITERION & OPTIMIZER
if args.backbone == "alexnet":
    model = torchvision.models.alexnet(pretrained=False).to(args.device)
if args.backbone == "resnet18":
    model = torchvision.models.resnet18(pretrained=False).to(args.device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

#### RUN EPOCHS
best_accuracy = 0
for epoch in range(args.epochs_num):
    #### TRAIN
    running_loss = torchmetrics.MeanMetric()
    for images, labels in train_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss.update(loss.item())
    
    #### VALIDATION
    accuracy = torchmetrics.Accuracy().cuda()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            accuracy.update(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
    
    accuracy = int(accuracy.compute().item() * 100)
    best_accuracy = max(accuracy, best_accuracy)
    logging.debug(f"Epoch: {epoch + 1:02d}/{args.epochs_num}; loss: {running_loss.compute().item():.3f}; " +
                  f"accuracy: {accuracy} %")

logging.info(f"Training took {str(datetime.now() - start_time)[:-7]}, best_accuracy: {best_accuracy}")

