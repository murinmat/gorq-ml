import torch
from numpy import ndarray
from collections import namedtuple


PerplexityResult = namedtuple('PerplexityResult', ['perplexity', 'counts'], )


def calc_perplexity(indices: torch.Tensor | ndarray, codebook_size: int) -> PerplexityResult:
    as_tensor = torch.as_tensor(indices)
    counts = torch.bincount(as_tensor.view(-1), minlength=codebook_size).float()
    probs = counts / counts.sum()
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum()
    perplexity = entropy.exp()

    return PerplexityResult(perplexity.item(), counts)
