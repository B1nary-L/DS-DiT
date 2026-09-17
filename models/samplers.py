import torch


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
def sample_euler(model, hr_noisy, lr, ref, sigmas, extra_args=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = hr_noisy.new_ones([hr_noisy.shape[0]])
    x = hr_noisy

    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, lr, ref, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, hr_noisy, lr, ref, sigmas, extra_args=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = hr_noisy.new_ones([hr_noisy.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    x = hr_noisy

    for i in range(len(sigmas) - 1):
        denoised = model(x, lr, ref, sigmas[i] * s_in, **extra_args)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x
