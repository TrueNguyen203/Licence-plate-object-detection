from datasets import load_dataset
import torch
import torchvision.transforms as T
from tools.lisence_plate_dataset import LisencePlateDataset
from tools.get_model import get_model
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
test_ds = ds['test']


def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))


transform = T.Compose

# create dataset
testing_dataset = LisencePlateDataset(
    data=test_ds,
    transform = T.ToTensor()
)

# create dataloader
test_loader = torch.utils.data.DataLoader(
    testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
)

# Load the model
model = get_model(num_classes = 2)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device)


if __name__ == "__main__":
    model.train()
    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # disable backprop but still allow loss computation
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass with targets gives loss dict
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())


            test_loss += losses.item()
            num_batches += 1
            if num_batches % 50 == 0:
                print("Loss dict:", {k: v.item() for k, v in loss_dict.items()}, "of batch: ", num_batches)

    avg_test_loss = test_loss / num_batches
    print(f"\nAverage test loss of batch: {avg_test_loss:.4f}")
