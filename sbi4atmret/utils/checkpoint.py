import torch


def load_checkpoint(path, device="cuda"):
    return torch.load(path, map_location=device)


def load_model_state(estimator, checkpoint):
    estimator.load_state_dict(checkpoint["estimator_state_dict"])


def save_checkpoint(path:str, estimator, optimizer=None, scheduler=None, epoch=0):
    torch.save({
        "epoch": epoch,
        "estimator_state_dict": estimator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, path)