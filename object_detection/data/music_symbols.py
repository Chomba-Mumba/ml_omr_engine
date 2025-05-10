transform = T.Compose([
    T.ToTensor(),
])

class musicDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
    
    def __getitem__(self,idx):
        #load image

        #load bounding boxes

        #load/define labels

        #Create target directory
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch, int64)

        #apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.data)