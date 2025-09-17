"""
U-Net. Implementation taken and modified from
https://github.com/mateuszbuda/brain-segmentation-pytorch

MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
# from .unet_layers import Conv1d, Conv2d

from torch.nn import Conv2d
from torch.nn import Parameter
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin

@dataclass
class UNetNormalOutput(UNet2DConditionOutput):
    def sample_noise_pred(self) -> torch.Tensor:
        noise_pred = self.sample[...,0] + self.sample[...,1] *  torch.randn_like(self.sample[..., 0], device=self.sample.device)
        return noise_pred

@dataclass
class UNetMvNormalOutput(UNet2DConditionOutput):
    def sample_noise_pred(self) -> torch.Tensor:
        predicted_noise_mu = self.sample[..., 0]
        diag = self.sample[..., 1]
        lora = self.sample[..., 2:]
        mvnorm = LowRankMultivariateNormal(
            predicted_noise_mu, lora, diag
        )
        noise_pred = mvnorm.sample()

        return noise_pred

@dataclass
class UNetMvMixedNormalOutput(UNet2DConditionOutput):
    def sample_noise_pred(self) -> torch.Tensor:
        mu = self.sample[..., 0]
        sigma = self.sample[..., 1]
        weights = self.sample[..., 2]
        sampled_weights = torch.distributions.Categorical(weights).sample()

        predicted_noise_mu = torch.gather(
            mu, dim=-1, index=sampled_weights.unsqueeze(-1)
        ).squeeze(-1)
        predicted_noise_sigma = torch.gather(
            sigma, dim=-1, index=sampled_weights.unsqueeze(-1)
        ).squeeze(-1)

        predicted_noise = predicted_noise_mu + predicted_noise_sigma * torch.randn_like(
            predicted_noise_mu, device=self.device
        )
        predicted_noise = predicted_noise.squeeze(-1)

        return predicted_noise


EPS = 1e-6


class UNetDiffusion(nn.Module):
    def __init__(
        self,
        d=1,
        conditioning_dim=3,
        hidden_channels=16,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cuda",
        domain_dim=128,
    ):
        super().__init__()
        self.d = d
        self.device = device
        self.hidden_dim = hidden_channels
        self.features = init_features
        self.input_projection = nn.Linear(
            in_channels, hidden_channels
        )  # the dimension of the target, is the dimension of the input of this MLP
        self.time_projection = nn.Linear(hidden_channels, hidden_channels)
        if d == 1:
            # self.unet = UNet1d(3 * hidden_channels, init_features)
            self.unet = SongUNet(
                img_resolution=domain_dim,
                in_channels=(in_channels + conditioning_dim),
                out_channels=1,
                d=1,
                attn_resolutions=[16],
                model_channels=hidden_channels,
            )
            self.output_projection = nn.Conv1d(
                in_channels=init_features,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            )
        elif d == 2:
            # self.unet = UNet2d(3 * hidden_channels, init_features)
            self.unet = SongUNet(
                img_resolution=domain_dim,
                in_channels=(in_channels + conditioning_dim),
                out_channels=1,
                d=2,
                attn_resolutions=[16],
                model_channels=hidden_channels,
            )
            self.output_projection = nn.Conv2d(
                in_channels=init_features,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            )
        else:
            raise NotImplementedError("Only 1D U-Net is implemented in this example.")

        if conditioning_dim:
            self.conditioning_projection = nn.Linear(conditioning_dim, hidden_channels)


    def forward_body(self, x_t, t, condition_input, **kwargs):
        # x_t and condition input have shape [B, C, D1,..., DN]
        t = t.unsqueeze(-1).type(torch.float32)
        x_t = self.unet(x_t, t, condition_input)
        return x_t

    def forward(self, x_t, t, condition_input, **kwargs):
        # x_t and condition_input have shape [B, C, D1,..., DN]
        # Forward body
        eps_pred = self.forward_body(x_t, t, condition_input)
        # Reprojection
        output = self.output_projection(eps_pred)
        return output


class UNet_diffusion_normal(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self, backbone, conv_out):
        super(UNet_diffusion_normal, self).__init__()
        # self.dtype = backbone.dtype
        
        self.backbone = backbone  # The UNet without the final conv_out layer
        in_channels = conv_out.in_channels
        out_channels = conv_out.out_channels
        
        # Prepare new weights and bias
        _weight = conv_out.weight.clone()  # [out_channels, in_channels, k, k]
        _bias = conv_out.bias.clone()

        self.mu_projection = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_out.kernel_size,
            stride=conv_out.stride,
            padding=conv_out.padding
        )
        
        self.mu_projection.weight = Parameter(_weight)
        self.mu_projection.bias = Parameter(_bias)
        
        self.sigma_projection = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_out.kernel_size,
            stride=conv_out.stride,
            padding=conv_out.padding
            )
        self.softplus = nn.Softplus()
        
        # replace config
        # backbone.config["out_channels"] = 2 * out_channels
        # logging.info("Unet config is updated")

    def forward(self, x_t, t, encoder_hidden_states, **kwargs):
        x_t = self.backbone(x_t, t, encoder_hidden_states).sample

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        sigma = self.softplus(sigma) + EPS
        output = torch.stack([mu, sigma], dim=-1)
        return UNetNormalOutput(output)


class UNet_diffusion_mvnormal(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self, backbone, conv_out, method="lora", rank=3):
        super(UNet_diffusion_mvnormal, self).__init__()
        self.backbone = backbone  # The UNet without the final conv_out layer
        self.in_channels = conv_out.in_channels
        self.out_channels = conv_out.out_channels

        self.method = method
        if method == "lora":
            sigma_out_channels = self.out_channels * (rank + 1)  # Rank + diagonal
        elif method == "cholesky":
            raise NotImplementedError("Cholesky is currently not implemented for image data")
            # # Covariance buffer
            # self.register_buffer(
            #     "tril_template",
            #     torch.zeros(self.domain_dim, self.domain_dim, dtype=torch.int64),
            # )
            # self.register_buffer(
            #     "tril_indices", torch.tril_indices(self.domain_dim, self.domain_dim)
            # )
            # self.tril_template[self.tril_indices.tolist()] = torch.arange(
            #     self.tril_indices.shape[1]
            # )
            # self.num_tril_params = self.tril_indices.shape[1]
            # sigma_out_channels = (self.domain_dim)//2 +1

        _weight = conv_out.weight.clone()  # [out_channels, in_channels, k, k]
        _bias = conv_out.bias.clone()

        self.mu_projection = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=conv_out.kernel_size,
            stride=conv_out.stride,
            padding=conv_out.padding,
        )
        
        self.mu_projection.weight = Parameter(_weight)
        self.mu_projection.bias = Parameter(_bias)
        
        self.sigma_projection = Conv2d(
            in_channels=self.in_channels,
            out_channels=sigma_out_channels,
            kernel_size=conv_out.kernel_size,
            stride=conv_out.stride,
            padding=conv_out.padding
        )
        self.softplus = nn.Softplus()

    def forward(self, x_t, t, encoder_hidden_states, **kwargs):
        x_t = self.backbone(x_t, t, encoder_hidden_states).sample

        mu = self.mu_projection(x_t).unsqueeze(-1)
        sigma = self.sigma_projection(x_t)
        if self.method == "lora":
            diag = self.softplus(sigma[:, 0 : self.out_channels]).unsqueeze(-1) + EPS
            lora = sigma[:, self.out_channels :,:,:]
            lora = lora.reshape(
                lora.shape[0],
                self.out_channels,
                -1,
                *lora.shape[2:]
            ).moveaxis(2, -1)
            lora = lora/(torch.norm(lora, dim=(-2,-1), keepdim=True) + EPS) # Normalize over only one of the image dimensions
            output = torch.cat([mu, diag, lora], dim=-1)

        elif self.method == "cholesky":
            raise NotImplementedError("Cholesky is currently not implemented for image data")
            # Initialize full zero matrix and fill lower triangle
            # L_full = torch.zeros(mu.shape[0], self.domain_dim, self.domain_dim, device=x_t.device)
            # L_full[:, self.tril_indices[0], self.tril_indices[1]] = sigma.flatten(start_dim = 1)[:,0:self.tril_indices[0].shape[0]]

            # # Enforce positive diagonal via softplus()
            # diag = nn.functional.softplus(torch.diagonal(L_full, dim1=-2, dim2=-1)) + EPS
            # L = torch.tril(L_full)
            # L = L/(torch.norm(L, dim=-1, keepdim=True) + EPS)
            # L[:, torch.arange(self.domain_dim), torch.arange(self.domain_dim)] = diag.squeeze()
            # L = L.unsqueeze(1)
            # output = torch.cat([mu, L], dim=-1)
        return UNetMvNormalOutput(output)


class UNet_diffusion_mixednormal(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self, backbone, conv_out, n_components=3):
        super(UNet_diffusion_mixednormal, self).__init__()
        self.backbone = backbone  # The UNet without the final conv_out layer
        self.in_channels = conv_out.in_channels
        self.out_channels = conv_out.out_channels

        self.n_components = n_components

        # Prepare new weights and bias
        _weight = conv_out.weight.clone()  # [out_channels, in_channels, k, k]
        _bias = conv_out.bias.clone()

        self.mu_projection = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * n_components,
            kernel=1,
        )

        # Repeat weights and bias for each mixture component        
        # self.mu_projection.weight = Parameter(_weight.repeat(1, n_components, 1, 1))
        # self.mu_projection.bias = Parameter(_bias.repeat(n_components))     
        
        
        
        self.sigma_projection = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * n_components,
            kernel=1,
            init_bias=1,
        )
        self.weights_projection = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * n_components,
            kernel=1,
        )

        self.softplus = nn.Softplus()
    
    
    
    def _initialize_weights(self, weight, bias):
        # Initialize weights and bias for mixture components with small perturbations; mean of weights should be the original weight
        weight = weight.repeat(1, self.n_components, 1, 1)
        bias = bias.repeat(self.n_components)
        
        k = 1 / torch.sqrt()
        
        for i in range(self.n_components):
            epsilon_weight = 1e-2 * torch.randn_like(weight)
            weight[i * self.out_channels : ((i + 1) % self.n_components) * self.out_channels] += epsilon_weight
            weight[((i+1) % self.n_components) * self.out_channels : ((i + 2) % self.n_components) * self.out_channels] -= epsilon_weight
            
            epsilon_bias = 1e-2 * torch.randn_like(bias)
            bias[i * self.out_channels : ((i + 1) % self.n_components) * self.out_channels] += epsilon_bias
            bias[((i+1) % self.n_components) * self.out_channels : ((i + 2) % self.n_components) * self.out_channels] -= epsilon_bias
            
            

    def _reshape_util(self, x):
        x = x.reshape(
            x.shape[0], self.out_channels, self.n_components, *x.shape[2:]
        ).moveaxis(2, -1)

        return x

    def forward(self, x_t, t, encoder_hidden_state, **kwargs):
        x_t = self.backbone.forward_body(x_t, t, encoder_hidden_state)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        weights = self.weights_projection(x_t)
        # Reshape
        mu = self._reshape_util(mu)
        sigma = self._reshape_util(sigma)
        weights = self._reshape_util(weights)

        # Apply postprocessing
        sigma = self.softplus(torch.clamp(sigma, min=-15)) + EPS
        weights = torch.softmax(torch.clamp(weights, min=-15, max=15), dim=-1)

        output = torch.stack([mu, sigma, weights], dim=-1)
        return output


if __name__ == "__main__":
    input = torch.randn(8, 1, 128)
    condition = torch.randn(8, 3, 128)
    output = torch.randn(8, 1, 128)
    t = torch.ones(8) * 0.5

    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cpu",
        domain_dim=(1,128),
    )
    unet = UNet_diffusion_mvnormal(backbone, d=1, target_dim=1, domain_dim = (1,128), method = "cholesky", rank = 5)
    test = unet.forward(input, t, condition)
    print(test.shape)