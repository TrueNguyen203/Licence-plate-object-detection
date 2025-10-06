import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision

class LisencePlateDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, index):
      image = self.data[index]['image']
      xmin = self.data[index]['objects']["bbox"][0][0]
      ymin = self.data[index]['objects']["bbox"][0][1]
      xmax = xmin + self.data[index]['objects']["bbox"][0][2]
      ymax = ymin + self.data[index]['objects']["bbox"][0][3]
      bbox = [xmin, ymin, xmax, ymax]
      # Access individual elements within the lists in the 'objects' dictionary
      target = {
          "boxes": torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0),
          "labels": torch.ones(1, dtype=torch.int64), # Labels should be a tensor of integers, one for each box
          "image_id": torch.tensor(self.data[index]['image_id']), # image_id should be a tensor
          "area": torch.as_tensor(self.data[index]['objects']["area"], dtype=torch.float32),
          "iscrowd": torch.zeros(1, dtype=torch.int64), # iscrowd should be a tensor of integers, one for each box
      }

      # Convert the image to a PIL Image before applying transformations

      if self.transform is not None:
          image = self.transform(image)

      return image, target
    