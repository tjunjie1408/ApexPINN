"""Exact equal-weight physics ensembles and explicit, finite no-window ablation parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from apexpinn.contracts import PredictionOutput
from apexpinn.models.stints import Scenario, StintHistory
from apexpinn.physics.residuals import PhysicsParams
from apexpinn.train.cloud_checkpoint import tree_digest


class NoWindowPhysics(PhysicsParams):
    """Gamma=0 exactly, with finite raw parameters and unchanged thermal/wear equations."""

    @property
    def gamma(self):
        return jnp.zeros_like(self.raw_gamma)


@dataclass
class PhysicsMember:
    model: Any
    constant_q: float | None = None

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        if self.constant_q is not None:
            history = replace(
                history,
                q=np.full_like(history.q, self.constant_q),
                ctx=replace(history.ctx, circuit_q_mean=self.constant_q),
            )
            scenario = replace(scenario, q_bar=self.constant_q)
        return self.model.forecast(history, horizon, scenario)

    def state_digest(self) -> str:
        return tree_digest(self.model) + ":" + str(self.constant_q)


@dataclass(frozen=True)
class PhysicsEnsemble:
    members: tuple[PhysicsMember, ...]

    def state_digest(self) -> str:
        from apexpinn.cloud.artifacts import identity

        return identity([m.state_digest() for m in self.members])

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        if len(self.members) != 5:
            raise ValueError("the declared physics ensemble has five members")
        predictions = [m.forecast(history, horizon, scenario) for m in self.members]
        means, scales = np.stack([p.mean_s for p in predictions]), np.stack([p.sigma_s for p in predictions])
        return PredictionOutput(
            mean_s=means.mean(axis=0),
            member_means_s=means,
            sigma_s=np.sqrt(np.mean(scales**2, axis=0) + np.var(means, axis=0)),
            member_sigma_s=scales,
            latent_h=np.stack([p.latent_h for p in predictions]).mean(axis=0),
            latent_w_s=np.stack([p.latent_w_s for p in predictions]).mean(axis=0),
        )
