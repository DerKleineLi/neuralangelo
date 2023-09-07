import torch


@torch.no_grad()
def aggregate_gradients(grad):
    sum = 0
    count = 0
    for gradient in grad:
        if gradient is not None:
            sum += gradient.data.abs().sum()
            count += gradient.data.numel()
    return (sum / count).detach().cpu().item()
