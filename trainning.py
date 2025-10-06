from datasets import load_dataset
import torch
import torchvision.transforms as T

from tools.lisence_plate_dataset import LisencePlateDataset
from tools.get_model import get_model
from tools.avg_loss import Averager
from pathlib import Path

# Set up agnostic device code
device = 'cuda' if torch.cuda.is_available else 'cpu'

# Setup model path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "model-with-anchor-nms.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Setup batch size
BATCH_SIZE = 4

# Loading the dataset from hugging face
ds = load_dataset("keremberke/license-plate-object-detection", "full")

train_ds = ds['train']
val_ds = ds['validation']

def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))


transform = T.Compose

# create dataset
training_dataset = LisencePlateDataset(
    data=train_ds,
    transform = T.ToTensor()
)
validation_dataset = LisencePlateDataset(
    data = val_ds,
    transform = T.ToTensor()
)

# create dataloader
train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate,
)



# Initialize model:
num_classes = 2  # background + license_plate
weights = "COCO_V1"
model = get_model(num_classes=num_classes, nms_thresh=0.3, weights = weights)

if device == 'cuda':
  model.to(device)

# Contruct loss, optimizer, and learning rate scheduler
train_losses = Averager()
val_losses = Averager()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lowest_val_loss = float('inf')
NUM_EPOCHS = 6


if __name__ == "__main__":
    # Train the model
    model.train()
    for epoch in range(NUM_EPOCHS):
        train_losses.reset()
        val_losses.reset()

        for batch_index, (images, targets) in enumerate(train_loader):
            # move the images and targets to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images,targets)
            loss = sum(loss for loss in loss_dict.values())

            # track the loss
            train_losses.send(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print(f"Epoch: {epoch} Batch Index: {batch_index} Loss: {loss.item()}")

        # evaluate
        with torch.no_grad():
            for _, (images, targets) in enumerate(validation_loader):
                # move the images and targets to device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                val_loss_dict = model(images, targets)
                val_loss = sum(loss for loss in val_loss_dict.values())

                # track the loss
                val_losses.send(val_loss.item())

        if val_losses.value < lowest_val_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            if lr_scheduler is not None:
                lr_scheduler.step()

        # print stats
        print(f"Epoch #{epoch} TRAIN LOSS: {train_losses.value} VALIDATION LOSS: {val_losses.value}\n")