
import sys
import torch
import logging
import argparse
import torchvision
import torchmetrics
from datetime import datetime
import torchvision.transforms as T
from torch.utils.data import Subset

import loss
import models
import commons

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", type=float, default=0.01, help="_")
parser.add_argument("--model", type=str, default="r9", choices=["r9", "r18"], help="_")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
parser.add_argument("--batch_size", type=int, default=64, help="_")
parser.add_argument("--num_workers", type=int, default=3, help="_")
parser.add_argument("--epochs_num", type=int, default=100, help="_")
parser.add_argument("--seed_weights", type=int, default=0, help="_")
parser.add_argument("--seed_optimization", type=int, default=0, help="_")

args = parser.parse_args()
start_time = datetime.now()
output_folder = f"logs/sw_{args.seed_weights:02d}__so_{args.seed_optimization:02d}"
commons.make_deterministic(args.seed_optimization)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### DATASETS & DATALOADERS
transform = T.Compose([
        T.ToTensor(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(32, scale=(0.8, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
if args.model == "r18":
    model = torchvision.models.resnet18(pretrained=False)
elif args.model == "r9":
    model = models.build_network()

model.load_state_dict(torch.load(f"models_{args.model}/{args.seed_weights:02d}.pth"))
model = model.to(args.device)

criterion = torch.nn.CrossEntropyLoss()
criterion2 = loss.CrossEntropyLabelSmooth(num_classes=10, epsilon=0.2)
if args.model == "r18":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True, weight_decay=0.001)

def lr(e):
    if e < 4:
        return 0.5*e/3. + 0.01
    return 0.5*(22-e)/19. + 0.01

sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)

#### RUN EPOCHS
best_val_accuracy = 0
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
    sched.step()
    
    #### VALIDATION
    val_accuracy = torchmetrics.Accuracy().cuda()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            val_accuracy.update(outputs, labels)
    
    val_accuracy = val_accuracy.compute().item() * 100
    if val_accuracy > best_val_accuracy:
        torch.save(model.state_dict(), f"{output_folder}/best_model.pth")
        best_val_accuracy = val_accuracy
    logging.debug(f"Epoch: {epoch + 1:02d}/{args.epochs_num}; " +
                  f"loss: {running_loss.compute().item():.3f}; " +
                  f"val_accuracy: {val_accuracy:.2f}%; " +
                  f"best_val_accuracy: {best_val_accuracy:.2f}%")

#### TEST with best model
model.load_state_dict(torch.load(f"{output_folder}/best_model.pth"))

test_accuracy = torchmetrics.Accuracy().cuda()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        outputs = model(images)
        test_accuracy.update(outputs, labels)

test_accuracy = test_accuracy.compute().item() * 100

logging.info(f"Training took {str(datetime.now() - start_time)[:-7]}; " +
             f"best_val_accuracy: {best_val_accuracy:.2f}; " +
             f"test_accuracy: {test_accuracy:.2f}")

