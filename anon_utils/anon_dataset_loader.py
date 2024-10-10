import torch

class AugDataset2(torch.utils.data.Dataset):
    def __init__(self, aug_dataset, original_dataset):
        self.x = aug_dataset
        self.x = self.x.float()

        print("set size:", self.x.shape)

        self.y = torch.IntTensor([data[1] for data in original_dataset])
        self.y = self.y.long()

        self.n_samples = aug_dataset.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]