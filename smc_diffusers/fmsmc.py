import torch
import numpy as np
from typing import Callable, Optional, Tuple
import logging

import os
import torchvision.utils as tvutils


class FMSMC:
    def __init__(
        self,
        *,
        lmbda: float,
        num_particles: int,
        rejuvenation: bool,
        naive_noise: bool,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        #reward_min_value: float = float("-inf"),
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device('cuda'),
        save_dir: str = None,
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"Unused arguments: {kwargs}")

        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps
        self.rejuvenation = rejuvenation
        self.naive_noise = naive_noise
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.log_product_of_potentials = torch.zeros(self.num_particles).to(self.device)

        # Lightweight inference accounting. These counters do not affect sampling.
        self.resampling_events = 0
        self.reward_evaluations = 0
        self.vae_decode_evaluations = 0



    def resample(
        self, *, 
        sampling_idx: int, 
        latents: torch.Tensor, 
        x0_preds: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )

        if sampling_idx not in resampling_interval:
            return latents, None

        self.resampling_events += 1
        self.vae_decode_evaluations += int(latents.shape[0])
        population_images = self.latent_to_decode_fn(x0_preds)
        rs_candidates = self.reward_fn(population_images)
        self.reward_evaluations += int(len(rs_candidates))

        if self.save_dir is not None:
            tvutils.save_image(population_images.data, f"images/{sampling_idx:02d}.png", nrow=4, normalize=True, value_range=(-1, 1))

        # lam = min(sampling_idx / self.resampling_t_end, 1)
        # Compute importance weights
        rs_candidates = torch.max(rs_candidates, self.population_rs)
        log_w = self.lmbda * rs_candidates

        if sampling_idx == self.resampling_t_end:
            log_w = self.lmbda * rs_candidates - self.log_product_of_potentials

        # def resample_latents(latents, indices):
        #     if not self.rejuvenation:
        #         return latents[indices]

        #     indices = indices.tolist()
        #     sampled_latents = []
        #     seen = set()
        #     for idx in indices:
        #         if idx not in seen:
        #             latent = latents[idx].clone()
        #             seen.add(idx)
        #         else:
        #             noise = torch.randn_like(latents[idx])
        #             latent = (1 - sigma) * x0_preds[idx] + sigma * noise
        #         sampled_latents.append(latent)
        #     sampled_latents = torch.stack(sampled_latents, dim=0)
        #     return sampled_latents

        def resample_latents(latents, indices):
            if not self.rejuvenation:
                return latents[indices]

            noise = torch.randn_like(latents)
            if self.naive_noise:
                #return latents[indices] + 0.5 * sigma * noise
                return latents[indices] + sigma * noise

            sampled_latents = (1 - sigma) * x0_preds[indices] + sigma * noise
            return sampled_latents

        normalized_w = torch.softmax(log_w, dim=0)
        if self.adaptive_resampling:
            # compute effective sample size
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                # print(f"Resampling at timestep {sampling_idx} with ESS: {ess}, normalized_w: ", normalized_w)
                # Resample indices based on weights
                indices = torch.multinomial(
                    normalized_w, num_samples=self.num_particles, replacement=True
                )
                # resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]
                resampled_latents = resample_latents(latents, indices)

                # Resample population images
                resampled_images = population_images[indices]

                # Update product of potentials; used for max and add potentials
                self.log_product_of_potentials = (
                    self.log_product_of_potentials[indices] + log_w[indices]
                )
            else:
                # print(f"No resampling! Timestep {sampling_idx} with ESS: {ess}, normalized_w: ", normalized_w)
                # No resampling
                resampled_images = population_images
                if self.rejuvenation:
                    noise = torch.randn_like(latents)
                    resampled_latents = (1 - sigma) * x0_preds + sigma * noise
                else:
                    resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                normalized_w, num_samples=self.num_particles, replacement=True
            )
            # resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]
            resampled_latents = resample_latents(latents, indices)

            # Resample population images
            resampled_images = population_images[indices]

            # Update product of potentials; used for max and add potentials
            self.log_product_of_potentials = (
                self.log_product_of_potentials[indices] + log_w[indices]
            )

        return resampled_latents, resampled_images
