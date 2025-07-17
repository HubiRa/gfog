import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class Domain:
    lower: torch.Tensor
    upper: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.all(torch.ge(x, self.lower)) and torch.all(torch.le(x, self.upper))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.lower, self.upper)


class TestFunctionBase(ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.domain: Optional[Domain] = None
        self.known_minima: Optional[List[torch.Tensor]] = None

    @abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Evaluate function on input tensor.
        Input: (..., input_dim) - any batch dimensions + feature dimension
        Output: (...) - same batch dimensions, scalar output
        """
        pass

    def __call__(self, x: torch.Tensor) -> List[float]:
        result = self.f(x)
        return result.detach().cpu().numpy().tolist()

    def _validate_and_project(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        if self.domain is not None:
            x = self.domain.project(x)

        return x

    def diff_from_minima(self, x: torch.Tensor) -> torch.Tensor:
        if self.known_minima is None:
            raise ValueError("No known minima for this function")

        x = self._validate_and_project(x)

        distances = []
        for minimum in self.known_minima:
            dist = torch.norm(x - minimum, dim=-1)
            distances.append(dist)

        return torch.min(torch.stack(distances), dim=0)[0]

    def set_domain(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        if lower.shape != (self.input_dim,) or upper.shape != (self.input_dim,):
            raise ValueError(f"Domain bounds must have shape ({self.input_dim},)")
        self.domain = Domain(lower, upper)

    def get_plot_ranges(self, n_points: int = 100):
        """Generate plot ranges based on function domain for 2D plotting"""
        if self.input_dim != 2:
            raise ValueError("Plot ranges only supported for 2D functions")
        if self.domain is None:
            raise ValueError("Domain must be set before generating plot ranges")

        from types import SimpleNamespace

        return SimpleNamespace(
            x=torch.linspace(self.domain.lower[0], self.domain.upper[0], n_points),
            y=torch.linspace(self.domain.lower[1], self.domain.upper[1], n_points),
        )


class BealeFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-4.5, -4.5]), upper=torch.tensor([4.5, 4.5])
        )

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return (
            (1.5 - x_val + x_val * y_val) ** 2
            + (2.25 - x_val + x_val * y_val**2) ** 2
            + (2.625 - x_val + x_val * y_val**3) ** 2
        )


class Rosenbrock2DFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-5.0, -5.0]), upper=torch.tensor([5.0, 5.0])
        )
        self.known_minima = [torch.tensor([1.0, 1.0])]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return (1 - x_val) ** 2 + 100 * (y_val - x_val**2) ** 2


class AckleyFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-5.0, -5.0]), upper=torch.tensor([5.0, 5.0])
        )
        self.known_minima = [torch.tensor([0.0, 0.0])]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return (
            -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x_val**2 + y_val**2)))
            - torch.exp(
                0.5 * (torch.cos(2 * np.pi * x_val) + torch.cos(2 * np.pi * y_val))
            )
            + np.e
            + 20
        )


class GoldsteinPriceFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-2.0, -2.0]), upper=torch.tensor([2.0, 2.0])
        )
        self.known_minima = [torch.tensor([0.0, -1.0])]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return (
            1
            + ((x_val + y_val + 1) ** 2)
            * (
                19
                - 14 * x_val
                + 3 * x_val**2
                - 14 * y_val
                + 6 * x_val * y_val
                + 3 * y_val**2
            )
        ) * (
            30
            + ((2 * x_val - 3 * y_val) ** 2)
            * (
                18
                - 32 * x_val
                + 12 * x_val**2
                + 48 * y_val
                - 36 * x_val * y_val
                + 27 * y_val**2
            )
        )


class HimmelblauFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-5.0, -5.0]), upper=torch.tensor([5.0, 5.0])
        )
        # Himmelblau's function has 4 global minima
        self.known_minima = [
            torch.tensor([3.0, 2.0]),
            torch.tensor([-2.805118, 3.131312]),
            torch.tensor([-3.779310, -3.283186]),
            torch.tensor([3.584428, -1.848126]),
        ]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return (x_val**2 + y_val - 11) ** 2 + (x_val + y_val**2 - 7) ** 2


class ThreeHumpCamelFunction(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(
            lower=torch.tensor([-5.0, -5.0]), upper=torch.tensor([5.0, 5.0])
        )
        self.known_minima = [torch.tensor([0.0, 0.0])]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        return 2 * x_val**2 - 1.05 * x_val**4 + x_val**6 / 6 + x_val * y_val + y_val**2


class MishrasBirdFunctionConstraint(TestFunctionBase):
    def __init__(self):
        super().__init__(input_dim=2)
        self.set_domain(lower=torch.tensor([-10, -6.5]), upper=torch.tensor([5, 5]))
        self.known_minima = [torch.tensor([-3.1302468, -1.5821422])]

    def _get_constraint(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # contraint: (x + 5)**2 + (y + 5)**2 - 25 < 0
        # we return the how much this constraint is violated
        violation = (x + 5) ** 2 + (y + 5) ** 2 - 25
        return torch.max(torch.zeros(len(x)), violation)

    def f(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._validate_and_project(x)
        x_val = x[..., 0]
        y_val = x[..., 1]
        fx = (
            torch.sin(y_val) * torch.exp((1 - torch.cos(x_val) ** 2))
            + torch.cos(x_val) * torch.exp((1 - torch.sin(y_val)) ** 2)
            + (x_val - y_val) ** 2
        )
        constraint = self._get_constraint(x_val, y_val)
        return {"fx": fx, "constraints": constraint}
