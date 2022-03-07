import torch
import torch.nn.functional as F


def smooth_crossentropy(
    pred: torch.Tensor, gold: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
