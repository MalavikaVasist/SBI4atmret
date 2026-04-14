import torch.optim as optim
import torch.optim.lr_scheduler as sched
from lampe.utils import GDStep
from zuko.distributions import BoxUniform

from .Loss import BNPELoss
from ..utils.load_utils import load_callable
from lampe.inference import NPELoss


def _get_config_value(config, *keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _find_config_section(config, *paths):
    for path in paths:
        if not path:
            continue
        value = _get_config_value(config, *path)
        if value is not None:
            return value
    return None


def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _parameter_bounds(config):
    parameter_list = _find_config_section(config, ("PARAMETERS",), ("Prior",), ("prior",))
    if parameter_list is None:
        raise KeyError("No PARAMETERS or Prior section found in config")
    lower = [p[1] for p in parameter_list]
    upper = [p[2] for p in parameter_list]
    return lower, upper


def _optimizer_config(config):
    return _find_config_section(config, ("training", "optimizer"), ("Loss", "optimizer"), ("optimizer",))


def _scheduler_config(config):
    return _find_config_section(config, ("training", "scheduler"), ("Loss", "scheduler"), ("scheduler",))


def _get_model_configs(config):
    return _find_config_section(config, ("ML_model_configs",), ("MLmodel_config",), ("model_configs",), ("model_config",)) or {}


def _get_loss_name(config, index):
    loss_section = _find_config_section(config, ("model_configs", "loss"), ("ML_model_configs", "loss"), ("MLmodel_config", "loss"), ("Loss", "loss_type"), ("Loss", "loss"), ("loss",))
    if loss_section is None:
        raise KeyError("Loss configuration not found")
    return _select_by_index(loss_section, index)


def setup_estimator(config, i: int):
    model_configs = _get_model_configs(config)
    estimator_cfg = _find_config_section(config, ("estimator",), ("ML_model_configs", "estimator"), ("MLmodel_config", "estimator"), ("model_configs", "estimator"))
    if estimator_cfg is None:
        raise KeyError("Estimator configuration not found")

    estimator_class = load_callable(estimator_cfg["module"], estimator_cfg["class"])
    lower, upper = _parameter_bounds(config)

    estimator = estimator_class(
        hf_miri=_select_by_index(model_configs["embedding"]["miri"], i),
        hf_inst=_select_by_index(model_configs["embedding"]["gemini"], i),
        instrument=estimator_cfg["instrument"],
        hidden_features=_select_by_index(model_configs["hidden_features"], i),
        emb_miri_output=_select_by_index(model_configs["embedding"]["miri_output"], i),
        emb_inst_output=_select_by_index(model_configs["embedding"]["gemini_output"], i),
        no_of_params=_select_by_index(model_configs["no_of_params"], i),
        transforms=_select_by_index(model_configs["transforms"], i),
        signal=_select_by_index(model_configs["signal"], i),
        LOWER=lower,
        UPPER=upper,
    ).cuda()
    return estimator


def _build_optimizer(estimator, optimizer_cfg):
    optimizer_type = optimizer_cfg.get("type")
    if optimizer_type is None:
        raise ValueError("Optimizer type must be provided in config")

    lr = optimizer_cfg.get("lr", optimizer_cfg.get("init_lr", 1e-3))
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    momentum = optimizer_cfg.get("momentum", 0.0)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    eps = optimizer_cfg.get("eps", 1e-8)

    if optimizer_type.lower() == "adamw":
        return optim.AdamW(estimator.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if optimizer_type.lower() == "adam":
        return optim.Adam(estimator.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if optimizer_type.lower() == "sgd":
        return optim.SGD(estimator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_type.lower() == "rmsprop":
        return optim.RMSprop(estimator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
    if optimizer_type.lower() == "adagrad":
        return optim.Adagrad(estimator.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type.lower() == "adadelta":
        return optim.Adadelta(estimator.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    if optimizer_type.lower() == "adamax":
        return optim.Adamax(estimator.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    raise NotImplementedError(f"Unsupported optimizer type: {optimizer_type}")


def _build_scheduler(optimizer, scheduler_cfg, config, index):
    if scheduler_cfg is None:
        return None

    scheduler_type = scheduler_cfg.get("type")
    if scheduler_type is None:
        return None

    scheduler_type = scheduler_type.lower()
    if scheduler_type == "reducelronplateau":
        factor = scheduler_cfg.get("factor", 0.5)
        min_lr = _select_by_index(scheduler_cfg.get("min_lr", 0.0), index)
        patience = _select_by_index(scheduler_cfg.get("patience", 10), index)
        threshold = scheduler_cfg.get("threshold", 0.01)
        return sched.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            min_lr=min_lr,
            patience=patience,
            threshold=threshold,
            threshold_mode=scheduler_cfg.get("threshold_mode", "abs"),
        )

    if scheduler_type == "steplr":
        step_size = scheduler_cfg.get("step_size")
        gamma = scheduler_cfg.get("gamma", 0.1)
        if step_size is None:
            raise ValueError("StepLR requires step_size in scheduler config")
        return sched.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_type == "exponentiallr":
        gamma = scheduler_cfg.get("gamma", 0.99)
        return sched.ExponentialLR(optimizer, gamma=gamma)

    if scheduler_type == "cosineannealinglr":
        T_max = scheduler_cfg.get("T_max")
        if T_max is None:
            raise ValueError("CosineAnnealingLR requires T_max in scheduler config")
        return sched.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=scheduler_cfg.get("eta_min", 0.0))

    if scheduler_type == "cosineannealingwarmrestarts":
        T_0 = scheduler_cfg.get("T_0")
        if T_0 is None:
            raise ValueError("CosineAnnealingWarmRestarts requires T_0 in scheduler config")
        return sched.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=scheduler_cfg.get("T_mult", 1),
            eta_min=scheduler_cfg.get("eta_min", 0.0),
        )

    if scheduler_type == "onecyclerl":
        max_lr = scheduler_cfg.get("max_lr")
        total_steps = scheduler_cfg.get("total_steps")
        if max_lr is None or total_steps is None:
            raise ValueError("OneCycleLR requires max_lr and total_steps in scheduler config")
        return sched.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    raise NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")


def setup_optimizer_and_scheduler(estimator, config, i: int):
    optimizer_cfg = _optimizer_config(config)
    if optimizer_cfg is None:
        raise KeyError("Optimizer configuration not found")

    optimizer = _build_optimizer(estimator, optimizer_cfg)
    clip = _find_config_section(config, ("training", "clip_grad_norm"), ("clip_grad_norm",)) or 1.0
    step = GDStep(optimizer, clip=clip)
    scheduler_cfg = _scheduler_config(config)
    scheduler = _build_scheduler(optimizer, scheduler_cfg, config, i)
    return optimizer, step, scheduler


def setup_loss_and_prior(estimator, config, i: int):
    loss_name = _get_loss_name(config, i)
    lower, upper = _parameter_bounds(config)
    prior = BoxUniform(torch.tensor(lower).cuda(), torch.tensor(upper).cuda())
    if loss_name == 'NPELoss':
        loss = NPELoss(estimator)
    elif loss_name == 'BNPELoss':
        loss = BNPELoss(estimator, prior)
    else:
        raise NotImplementedError(f"Unsupported loss function: {loss_name}")
    return loss, prior


def setup_pipe(config, loss):
    pipe_func = load_callable(config["pipe"]["module"], config["pipe"]["function"])
    return pipe_func(loss)
