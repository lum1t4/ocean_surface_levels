import torch
from typing import Callable, Literal

# DDPM Paper https://arxiv.org/pdf/2006.11239.pdf
# https://web.archive.org/web/20240229175023/https://ben.bolte.cc/diffusion-flow-matching
# https://www.tonyduan.com/diffusion/index.html
# https://aman.ai/primers/ai/diffusion-models/


# ------------------------------
# Variance schedule B_1, ..., B_T
# ------------------------------

def cosine_beta_schedule(steps: int, s: float = 0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, steps)

def quadratic_beta_schedule(steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, steps) ** 2

def sigmoid_beta_schedule(steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    betas = torch.linspace(-6, 6, steps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract values from 1D tensor `a` at indices `t`, reshape for broadcasting to `x_shape`."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def _broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.reshape(b.shape[0], *((1,) * (len(b) - 1)))


def randn_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891 (Algorithm 1)

    Args:
        betas (`torch.Tensor`):
            The betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`:
            Rescaled betas with zero terminal SNR.
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


VarianceType = Literal["fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"]
PredictionType = Literal["epsilon", "sample", "velocity"]
     

class DDPMScheduler:
    def __init__(
        self,
        steps: int = 1000,
        schedule: Callable[..., torch.Tensor] = linear_beta_schedule,
        generator: torch.Generator | None = None,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        rescale_betas_zero_snr: bool = True,
        prediction_type: PredictionType = "epsilon",
        variance_type: VarianceType = "fixed_small",
        **kargs
    ):
        self.num_train_steps = steps
        self.generator = generator
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type
        self.variance_type = variance_type


        self.betas: torch.Tensor = schedule(steps) # variance schedule
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas: torch.Tensor = 1. - self.betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod: torch.Tensor = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod) ** 0.5
        self.timesteps = None
        self.set_inference_steps(self.num_train_steps)

    def set_inference_steps(self, num_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain.
        """
        self.num_infer_steps = num_steps
        self.step_ratio = self.num_train_steps // self.num_infer_steps
        self.timesteps = (torch.arange(num_steps) * self.step_ratio).flip(0)
    
    def previous_timestep(self, timesteps: int | torch.Tensor) -> int | torch.Tensor:
        return timesteps - self.step_ratio
    
    def forward_sample(self, x0: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward (noising) process q(x_t | x_0) of a Denoising Diffusion Probabilistic Model (DDPM)
        """

        # We have a closed formula to sample x_t from DDPM paper
        # without computing itermediate corrupted samples:
        # q(x_t | x_0) = N(sqrt(alpha_bar[t]) * x_0, (1 - alpha_bar[t]) * I)
        # Recall: Z ~ N(0, I) -> aZ + b ~ N(b, a^2, I)
        # x_t can be sampled as a linear combination of the original
        # x0 and standard normal noise:
        # x_t = sqrt(alphas_cp[t]) * x_0 + sqrt(beta_cp[t]) * ε

        # noise ε ~ N(0, I)
        noise = randn_like(x0, generator=self.generator)
        coeff_signal = extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        coeff_noise = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        xt = coeff_signal * x0 + coeff_noise * noise
        return xt, noise

    def set_strength(self, strength: float = 1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_infer_steps - int(self.num_infer_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def predict_x0(self, xt: torch.Tensor, prediction: torch.Tensor, acp_curr: torch.Tensor, bcp_curr: torch.Tensor) -> torch.Tensor:
        match self.prediction_type:
            case 'epsilon':
                # model predicts noise
                # Formula Eq (15) DDPM paper 
                return (xt - bcp_curr ** 0.5 * prediction) / acp_curr ** 0.5
            case 'sample':
                # model predicts original non-corrupted sample
                return prediction
            case 'velocity':
                # model predicts velocity
                return (acp_curr**0.5) * xt - (bcp_curr**0.5) * prediction


    def reverse_sample(self, xt: torch.Tensor, timesteps: int, pred: torch.Tensor, pred_variance: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.variance_type not in {"learned", "learned_range"} or pred_variance is not None, f"Predicted variance cannot be none if variance type is {self.variance_type}"

        curr = timesteps
        prev = self.previous_timestep(curr)

        # 1. Compute alphas, betas
        acp_curr = extract(self.alphas_cumprod, curr, xt.shape)
        acp_prev = torch.where(
            prev.reshape(-1, *([1] * (xt.ndim - 1))) >= 0,
            extract(self.alphas_cumprod, prev.clamp(min=0), xt.shape),
            torch.ones_like(acp_curr),
        )
        bcp_curr = 1. - acp_curr
        bcp_prev = 1. - acp_prev
        a_curr = acp_curr / acp_prev
        b_curr = 1. - a_curr

        # 2. Predict original sample
        x0_pred = self.predict_x0(xt, pred, acp_curr, bcp_curr)

        # 3. x0
        if self.clip_sample:
            x0_pred = x0_pred.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 5. Compute predicted previous sample mean µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        posterior_mean_coef1 = (acp_prev ** 0.5 * b_curr) / bcp_curr
        posterior_mean_coef2 = (a_curr ** 0.5 * bcp_prev) / bcp_curr
        posterior_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * xt
        
        # 6. Add noise
        noise = randn_like(xt, generator=self.generator)
        # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        variance = (1 - acp_prev) / (1 - acp_curr) * b_curr
        variance = variance.clamp(min=1e-20)

        match self.variance_type:
            case "fixed_small":
                variance = variance
                deviation = variance.sqrt()

            case "fixed_small_log":
                variance = torch.log(variance)
                deviation = torch.exp(0.5 * variance)
            
            case "fixed_large":
                variance = bcp_curr
                deviation = variance.sqrt()
            
            case "learned":
                variance = pred_variance
                deviation = variance.sqrt()

            case "learned_range":
                min_log = torch.log(variance)
                max_log = torch.log(bcp_curr)
                frac = (pred_variance + 1) / 2
                variance = frac * max_log + (1 - frac) * min_log
                deviation = torch.exp(0.5 * variance)
            case _:
                raise ValueError(f"Variance Type {self.variance_type} is not valid or not supported")
        
        # if t == 0: return mean (no noise)
        non_zero = (timesteps != 0).float().reshape(timesteps.shape[0], *((1,) * (xt.ndim - 1)))
        x_prev = posterior_mean + non_zero * deviation * noise
        return x_prev, x0_pred
    
    def to(self, device: torch.device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler.

    DDIM generalizes DDPM by defining a family of non-Markovian diffusion
    processes that share the same forward marginals as DDPM but allow for
    deterministic (eta=0) or stochastic (eta>0) reverse sampling with
    fewer steps.

    Key difference from DDPM:
    - The reverse process is parameterized by eta (η):
        - η = 0: fully deterministic (ODE)
        - η = 1: equivalent to DDPM
        - 0 < η < 1: interpolation between deterministic and stochastic
    """

    def __init__(
        self,
        steps: int = 1000,
        schedule: Callable[..., torch.Tensor] = linear_beta_schedule,
        generator: torch.Generator | None = None,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        rescale_betas_zero_snr: bool = True,
        prediction_type: PredictionType = "epsilon",
        eta: float = 0.0,
        **kwargs,
    ):
        self.num_train_steps = steps
        self.generator = generator
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type
        self.eta = eta

        self.betas: torch.Tensor = schedule(steps)
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas: torch.Tensor = 1.0 - self.betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod: torch.Tensor = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = (1.0 - self.alphas_cumprod) ** 0.5

        # For the final step (t=0), we treat alpha_cumprod_prev as 1.0
        self.final_alpha_cumprod = torch.tensor(1.0)

        self.timesteps = None
        self.set_inference_steps(self.num_train_steps)

    def set_inference_steps(self, num_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain.
        DDIM allows using fewer steps than training (e.g., 50 instead of 1000).
        """
        self.num_infer_steps = num_steps
        self.step_ratio = self.num_train_steps // self.num_infer_steps
        # Evenly spaced subsequence of [0, T), reversed for denoising
        self.timesteps = (torch.arange(num_steps) * self.step_ratio).flip(0)

    def previous_timestep(self, timesteps: int | torch.Tensor) -> int | torch.Tensor:
        return timesteps - self.step_ratio

    def forward_sample(
        self, x0: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward (noising) process — identical to DDPM since DDIM shares the
        same forward marginals: q(x_t | x_0) = N(sqrt(ā_t) x_0, (1 - ā_t) I)
        """
        noise = randn_like(x0, generator=self.generator)
        coeff_signal = extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        coeff_noise = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        xt = coeff_signal * x0 + coeff_noise * noise
        return xt, noise

    def set_strength(self, strength: float = 1.0):
        start_step = self.num_infer_steps - int(self.num_infer_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def predict_x0(
        self,
        xt: torch.Tensor,
        prediction: torch.Tensor,
        acp_curr: torch.Tensor,
        bcp_curr: torch.Tensor,
    ) -> torch.Tensor:
        match self.prediction_type:
            case "epsilon":
                return (xt - bcp_curr**0.5 * prediction) / acp_curr**0.5
            case "sample":
                return prediction
            case "velocity":
                return (acp_curr**0.5) * xt - (bcp_curr**0.5) * prediction

    def predict_epsilon(
        self,
        xt: torch.Tensor,
        prediction: torch.Tensor,
        acp_curr: torch.Tensor,
        bcp_curr: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the noise prediction regardless of prediction_type."""
        match self.prediction_type:
            case "epsilon":
                return prediction
            case "sample":
                return (xt - acp_curr**0.5 * prediction) / bcp_curr**0.5
            case "velocity":
                return (acp_curr**0.5) * prediction + (bcp_curr**0.5) * xt

    def reverse_sample(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        pred: torch.Tensor,
        pred_variance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM reverse step (Eq. 12 from the DDIM paper):

        x_{t-1} = sqrt(ā_{t-1}) * x0_pred
                + sqrt(1 - ā_{t-1} - σ²_t) * ε_pred
                + σ_t * z

        where:
            σ_t = η * sqrt((1 - ā_{t-1}) / (1 - ā_t)) * sqrt(1 - ā_t / ā_{t-1})

        When η = 0 the process is deterministic (no noise added).
        """
        curr = timesteps
        prev = self.previous_timestep(curr)

        # 1. Compute alpha cumprods
        acp_curr = extract(self.alphas_cumprod, curr, xt.shape)
        # For prev < 0 (i.e., t=0), use final_alpha_cumprod = 1.0
        acp_prev = torch.where(
            prev.reshape(-1, *([1] * (xt.ndim - 1))) >= 0,
            extract(self.alphas_cumprod, prev.clamp(min=0), xt.shape),
            self.final_alpha_cumprod.to(xt.device),
        )
        bcp_curr = 1.0 - acp_curr
        bcp_prev = 1.0 - acp_prev

        # 2. Predict x0 and epsilon
        x0_pred = self.predict_x0(xt, pred, acp_curr, bcp_curr)
        eps_pred = self.predict_epsilon(xt, pred, acp_curr, bcp_curr)

        # 3. Clip x0 if requested
        if self.clip_sample:
            x0_pred = x0_pred.clamp(-self.clip_sample_range, self.clip_sample_range)
            # Recompute eps from clipped x0 for consistency
            eps_pred = (xt - acp_curr**0.5 * x0_pred) / bcp_curr**0.5

        # 4. Compute sigma (DDIM Eq. 16)
        # σ_t = η * sqrt((1 - ā_{t-1}) / (1 - ā_t) * (1 - ā_t / ā_{t-1}))
        sigma = (
            self.eta
            * (bcp_prev / bcp_curr * (1.0 - acp_curr / acp_prev)).clamp(min=0).sqrt()
        )

        # 5. Compute "direction pointing to x_t" coefficient
        # sqrt(1 - ā_{t-1} - σ²)
        pred_direction_coeff = (bcp_prev - sigma**2).clamp(min=0).sqrt()

        # 6. Compute x_{t-1} (DDIM Eq. 12)
        x_prev = acp_prev**0.5 * x0_pred + pred_direction_coeff * eps_pred

        # 7. Add noise (only when η > 0 and t > 0)
        if self.eta > 0:
            noise = randn_like(xt, generator=self.generator)
            non_zero = (timesteps != 0).float().reshape(
                timesteps.shape[0], *((1,) * (xt.ndim - 1))
            )
            x_prev = x_prev + non_zero * sigma * noise

        return x_prev, x0_pred

    def to(self, device: torch.device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(device)
        return self
