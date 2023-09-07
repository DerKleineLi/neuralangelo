import torch


@torch.no_grad()
def aggregate_gradients(model):
    sum = 0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            sum += param.grad.data.abs().sum()
            count += param.grad.data.numel()
    return (sum / count).detach().cpu().item()
