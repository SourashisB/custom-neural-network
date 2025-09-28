# src/optim.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

Param = Dict[str, np.ndarray]


def _as_param_list(params: Iterable[Param]) -> List[Param]:
    # Materialize generator to list for state alignment
    pl = list(params)
    # Validate expected keys
    for i, p in enumerate(pl):
        if not isinstance(p, dict) or "param" not in p or "grad" not in p:
            raise TypeError(f"Parameter at index {i} must be a dict with 'param' and 'grad' ndarrays.")
        if not isinstance(p["param"], np.ndarray) or not isinstance(p["grad"], np.ndarray):
            raise TypeError(f"Parameter at index {i} 'param' and 'grad' must be np.ndarray.")
        if p["param"].shape != p["grad"].shape:
            raise ValueError(f"Parameter and grad shapes differ at index {i}: {p['param'].shape} vs {p['grad'].shape}")
    return pl


@dataclass
class SGD:
    """
    Stochastic Gradient Descent with momentum, optional Nesterov, and weight decay.

    - lr: learning rate
    - momentum: if > 0, uses classical momentum v = mu*v - lr*g
    - nesterov: if True, uses Nesterov accelerated gradient
    - weight_decay: L2 regularization (decoupled): param <- param - lr * wd * param

    Update rules (per parameter p):
      if weight_decay > 0:
        p <- p - lr * weight_decay * p        # decoupled weight decay

      g <- grad
      if momentum == 0:
        p <- p - lr * g
      else:
        v <- mu * v + g
        if nesterov:
          p <- p - lr * (mu * v + g)
        else:
          p <- p - lr * v
    """

    lr: float = 1e-2
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0

    # State
    _velocities: List[np.ndarray] = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)

    def _init_state(self, params: List[Param]) -> None:
        self._velocities = [np.zeros_like(p["param"]) for p in params]
        self._initialized = True

    def step(self, params_iter: Iterable[Param]) -> None:
        params = _as_param_list(params_iter)
        if not self._initialized:
            self._init_state(params)

        mu = float(self.momentum)
        wd = float(self.weight_decay)
        lr = float(self.lr)

        for i, p in enumerate(params):
            w = p["param"]
            g = p["grad"]

            # Decoupled weight decay
            if wd != 0.0:
                w[...] = w - lr * wd * w

            if mu == 0.0:
                # Vanilla SGD
                w[...] = w - lr * g
            else:
                v = self._velocities[i]
                # v = mu*v + g
                v[...] = mu * v + g
                if self.nesterov:
                    # p <- p - lr * (mu*v + g)
                    w[...] = w - lr * (mu * v + g)
                else:
                    # p <- p - lr * v
                    w[...] = w - lr * v

    def zero_state(self) -> None:
        self._velocities = []
        self._initialized = False


@dataclass
class Adam:
    """
    Adam optimizer with bias correction and decoupled weight decay.

    Parameters:
      - lr: learning rate (alpha)
      - betas: (beta1, beta2) momentum terms
      - eps: numerical stability term
      - weight_decay: L2 regularization (decoupled); AdamW-style if > 0

    Update (per parameter p):
      if weight_decay > 0:
        p <- p - lr * weight_decay * p        # decoupled weight decay

      m <- beta1*m + (1 - beta1)*g
      v <- beta2*v + (1 - beta2)*g^2
      m_hat <- m / (1 - beta1^t)
      v_hat <- v / (1 - beta2^t)
      p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    # State
    _m: List[np.ndarray] = field(default_factory=list, init=False)
    _v: List[np.ndarray] = field(default_factory=list, init=False)
    _t: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def _init_state(self, params: List[Param]) -> None:
        self._m = [np.zeros_like(p["param"]) for p in params]
        self._v = [np.zeros_like(p["param"]) for p in params]
        self._t = 0
        self._initialized = True

    def step(self, params_iter: Iterable[Param]) -> None:
        params = _as_param_list(params_iter)
        if not self._initialized:
            self._init_state(params)

        beta1, beta2 = self.betas
        beta1 = float(beta1)
        beta2 = float(beta2)
        lr = float(self.lr)
        eps = float(self.eps)
        wd = float(self.weight_decay)

        # Time step
        self._t += 1
        t = self._t

        # Precompute bias correction factors
        bias_c1 = 1.0 - beta1**t
        bias_c2 = 1.0 - beta2**t

        for i, p in enumerate(params):
            w = p["param"]
            g = p["grad"]

            # Decoupled weight decay
            if wd != 0.0:
                w[...] = w - lr * wd * w

            m = self._m[i]
            v = self._v[i]

            # Update moments
            m[...] = beta1 * m + (1.0 - beta1) * g
            v[...] = beta2 * v + (1.0 - beta2) * (g * g)

            # Bias-corrected moments
            m_hat = m / bias_c1
            v_hat = v / bias_c2

            # Parameter update
            w[...] = w - lr * (m_hat / (np.sqrt(v_hat) + eps))

    def zero_state(self) -> None:
        self._m = []
        self._v = []
        self._t = 0
        self._initialized = False