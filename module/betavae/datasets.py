from torch.utils.data import Sampler
import torch


class FilterSampler(Sampler):
    def __init__(self, targets, keep_letters, all_letters):
        true_classes = [i+1 for i, lett in enumerate(all_letters) if lett in keep_letters]
        self.mask = torch.tensor([(elem in true_classes) for elem in targets])
        self.indices = list(range(len(self.mask)))

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))
    
    def __len__(self):
        return sum(self.mask)
