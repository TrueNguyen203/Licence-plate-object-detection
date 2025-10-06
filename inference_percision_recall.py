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

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1, boxes2: (N, 4), (M, 4)
    Returns IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def compute_precision_recall(predictions, targets, iou_threshold=0.5):
    """
    Compute precision and recall considering ONLY the highest-score
    predicted bounding box per image.

    Args:
        predictions: list of dicts from model output, each with keys:
                     ['boxes', 'labels', 'scores']
        targets: list of dicts with keys:
                 ['boxes', 'labels']
        iou_threshold: float, IoU threshold to consider a true positive

    Returns:
        precision, recall
    """
    TP, FP, FN = 0, 0, 0

    for pred, target in zip(predictions, targets):
        gt_boxes = target['boxes'].detach().cpu()
        gt_labels = target['labels'].detach().cpu()

        # If there is at least one prediction, take only the highest score
        if len(pred['boxes']) > 0:
            scores = pred['scores'].detach().cpu()
            best_idx = torch.argmax(scores)
            pred_box = pred['boxes'][best_idx].detach().cpu().unsqueeze(0)
            pred_label = pred['labels'][best_idx].detach().cpu().unsqueeze(0)
        else:
            pred_box = torch.empty((0, 4))
            pred_label = torch.empty((0,), dtype=torch.long)

        if len(pred_box) == 0:
            # No prediction → all GTs are missed
            FN += len(gt_boxes)
            continue

        # Compute IoU between this single predicted box and all GT boxes
        ious = box_iou(pred_box, gt_boxes).squeeze(0)
        max_iou, max_idx = (ious.max(0) if len(ious) > 0 else (torch.tensor(0.0), torch.tensor(-1)))

        if max_iou >= iou_threshold and pred_label[0] == gt_labels[max_idx]:
            TP += 1
        else:
            FP += 1

        # If no match found → all GT boxes are considered missed except matched ones
        FN += len(gt_boxes) - (1 if max_iou >= iou_threshold else 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall

if __name__ == "__main__":
    model.to(device)
    model.eval()

    images = []
    targets = []
    predictions = []
    model.eval()
    for i in range(0,882):
        image, target = testing_dataset[i]
        images.append(image.to(device)) # Move image to the device
        targets.append(target)
        with torch.no_grad():
            # The model expects a list of images
            predictions.append(model([image.to(device)])[0]) # Move image to the device

    precision50, recall50 = compute_precision_recall(predictions, targets)
    precision75, recall75 = compute_precision_recall(predictions, targets, iou_threshold=0.75)
    print(f"Precision: {precision50:.4f}, Recall: {recall50:.4f}")
    print(f"Precision: {precision75:.4f}, Recall: {recall75:.4f}")
