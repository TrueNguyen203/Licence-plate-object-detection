from datasets import load_dataset
import torch
import torchvision.transforms as T

from pathlib import Path
from tools.lisence_plate_dataset import LisencePlateDataset
from tools.get_model import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_score, recall_score


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


if __name__ == "__main__":
    model.to(device)
    model.eval()

    # metric objects
    map_metric = MeanAveragePrecision()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # predictions on GPU
            predictions = model(images)

            # move preds + targets to CPU for torchmetrics
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            # update mAP metric
            map_metric.update(preds_cpu, targets_cpu)

            # collect labels for confusion matrix
            for pred, tgt in zip(preds_cpu, targets_cpu):
                all_preds.extend(pred["labels"].numpy())
                all_targets.extend(tgt["labels"].numpy())

    # compute mAP
    map_results = map_metric.compute()
    print("\nMean Average Precision (mAP):")
    print(map_results)
