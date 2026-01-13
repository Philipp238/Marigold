from itertools import product
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import marigold.diffusionUQ.losses as losses
import scoringrules as sr
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def get_criterion(training_parameters, beta=None):
    """Define criterion for the model.
    Criterion gets as arguments (truth, prediction) and returns a loss value.
    """
    # Hard-code lora for now, as Cholesky is not implemented
    method = "lora" # training_parameters["mvnormal_method"]
    loss = training_parameters["loss"]
    if training_parameters["uncertainty_quantification"] == "diffusion":
        if training_parameters["distributional_method"] == "deterministic":
            criterion = nn.MSELoss()
        elif training_parameters["distributional_method"] == "normal":
            if loss == "crps":
                criterion = lambda truth, prediction: sr.crps_normal(
                    truth, prediction[..., 0], prediction[..., 1], backend="torch"
                ).mean()

            elif loss == "kernel":
                criterion = losses.GaussianKernelScore(dimension = "univariate", gamma = training_parameters["gamma"])
            elif loss == "log":
                criterion = lambda truth, prediction: (-1)* Normal(prediction[...,0], prediction[...,1]).log_prob(truth).mean()
            else:
                raise AssertionError("Loss function not implemented")
        elif training_parameters["distributional_method"] == "iDDPM":
            criterion = losses.iDDPMLoss(
                beta = beta,
                loss_lambda= training_parameters["iDDPM_lambda"],
            )
        elif training_parameters["distributional_method"] == "sample":
            criterion = lambda truth, prediction: sr.energy_score(
                truth.flatten(start_dim=1, end_dim=-1),
                prediction.flatten(start_dim=1, end_dim=-2),
                m_axis=-1,
                v_axis=-2,
                backend="torch",
            ).mean()
        elif training_parameters["distributional_method"] == "mixednormal":
            criterion = losses.NormalMixtureCRPS()
        elif training_parameters["distributional_method"] == "mvnormal":
            if method == "lora":
                if loss == "kernel":
                    criterion = losses.GaussianKernelScore(dimension = "multivariate", gamma = training_parameters["gamma"], method = "lora")
                elif loss == "log":
                    criterion = lambda truth, prediction: (-1)* LowRankMultivariateNormal(prediction[...,0], prediction[...,2:], prediction[...,1]).log_prob(truth).mean()
            elif method == "cholesky":
                if loss == "log":
                    criterion = lambda truth, prediction: (-1)* MultivariateNormal(loc = prediction[...,0], scale_tril=prediction[...,1:]).log_prob(truth).mean()
                elif loss == "kernel":
                    criterion = losses.GaussianKernelScore(dimension = "multivariate", gamma = training_parameters["gamma"], method = "cholesky")
        else:
            raise ValueError(
                f'"distributional_method" must be any of the following: "deterministic", "normal", "sample" or'
                f'"mixednormal". You chose {training_parameters["distributional_method"]}.'
            )
    else:
        criterion = nn.MSELoss()
    return criterion