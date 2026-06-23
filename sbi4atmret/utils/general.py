



def transform_uniform(x, a, b, c, d):
        # Check if x is within the original range
        if a <= x <= b:
            # Apply the transformation formula
            y = c + ((x - a) * (d - c)) / (b - a)
            return y
        else:
            raise ValueError(f"x must be in the range [{a}, {b}]")




def instrument_from_simname(sim_name: str) -> str:
    """
    Extract instrument name from a sim_name key.

    Examples:
        "cloudfree_miri"   -> "miri"
        "cloudfree_gemini" -> "gemini"
        "cloudy_hst"       -> "hst"
    """
    return sim_name.split("_", 1)[1]


def find_map_sample(net, theta, x_obs, device="cuda", batch_size=1000):
    """
    Find the Maximum A Posteriori (MAP) sample from theta
    by evaluating log-probability under the posterior.

    Args:
        net: model with .flow(x) returning a posterior distribution
        theta: (N, D) tensor of posterior samples
        x_obs: (1, D_obs) tensor of the observation (already on device)
        device: device for computation
        batch_size: process theta in batches to avoid OOM

    Returns:
        (index, sample) — index into theta and the MAP sample as a 1D tensor
    """
    import torch

    posterior = net.flow(x_obs.to(device))

    max_log_p = float("-inf")
    max_index = -1

    with torch.no_grad():
        for i in range(0, len(theta), batch_size):
            torch.cuda.empty_cache()

            theta_batch = theta[i : i + batch_size].float().to(device)
            log_p_batch = posterior.log_prob(theta_batch).cpu()

            # Filter non-finite values
            finite_mask = torch.isfinite(log_p_batch)
            log_p_finite = log_p_batch[finite_mask]

            if log_p_finite.numel() == 0:
                continue

            batch_max = log_p_finite.max().item()

            if batch_max > max_log_p:
                max_log_p = batch_max
                local_idx = log_p_finite.argmax().item()
                # Map back to global index
                max_index = i + torch.where(finite_mask)[0][local_idx].item()

    map_sample = theta[max_index]

    return max_index, map_sample
