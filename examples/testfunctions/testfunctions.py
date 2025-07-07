import torch
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from gopt.utils import maybe_unsqueeze


class InputLayout:
    input_layouts = [
        "X",
        "BX",
        "XY",
        "BXY",
        "XYZ",
        "BXYZ",
    ]
    input_layouts = input_layouts + [f"{il}_SHAPE" for il in input_layouts]

    def __init__(self) -> None:
        self.X: int = 1
        self.X_SHAPE: int = 1
        self.BX: int = 2
        self.BX_SHAPE: int = 1
        self.XY: int = 1
        self.XY_SHAPE: int = 2
        self.BXY: int = 2
        self.BXY_SHAPE: int = 2
        self.XYZ: int = 1
        self.XYZ_SHAPE: int = 3
        self.BXYZ: int = 2
        self.BXYZ_SHAPE: int = 3

    def __getitem__(self, key) -> int:
        if key is None:
            return None
        else:
            assert key in self.input_layouts
            return self.__dict__[key]

    def has_batch_dim(self, input_layout: str) -> bool:
        assert input_layout in self.input_layouts
        return "B" in input_layout

    def _get_plot_info(self, input_layout: str) -> dict:
        assert input_layout in self.input_layouts
        dim = self[input_layout]
        multisamples = self.has_batch_dim(input_layout)
        input_dim = self[input_layout + "_SHAPE"]
        return {
            "total_dim": dim,
            "input_dim": input_dim,
            "multisamples": multisamples,
        }


@dataclass
class Domain:
    lower: torch.Tensor
    upper: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return all(torch.le(self.lower, x).view(-1)) and all(
            torch.le(x, self.upper).view(-1)
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.lower, self.upper)

    def get_lower(self) -> torch.Tensor:
        return self.lower.flatten()

    def get_upper(self) -> torch.Tensor:
        return self.upper.flatten()


class TestFunctionBase(ABC):
    layout_info: InputLayout = InputLayout()

    def __init__(self, input_layout: str):
        self.input_layout: str = input_layout
        self.input_layout_dim: int = self.layout_info[input_layout]
        self.domain: Domain = None
        known_minima: list[torch.Tensor] = None

    @abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, vec: torch.Tensor) -> list[float]:
        val = self.f(vec)
        val = list(val.detach().cpu().numpy())
        return val

    def _check_and_maybe_unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > self.input_layout_dim:
            raise ValueError(
                f"Input tensor has {x.dim()} dimensions, but expected dim <= {self.input_layout_dim}"
            )
        elif x.dim() < self.input_layout_dim - 1 and self.layout_info.has_batch_dim(
            self.input_layout
        ):
            raise ValueError(
                f"Input tensor with dimension {x.dim()} is underdefined for layout {self.input_layout}"
            )
        elif x.dim() < self.input_layout_dim and not self.layout_info.has_batch_dim(
            self.input_layout
        ):
            raise ValueError(
                f"Input tensor with dimension {x.dim()} is underdefined for layout {self.input_layout}"
            )

        x = maybe_unsqueeze(x, self.input_layout_dim)
        return x

    def diff_from_minima(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_and_maybe_unsqueeze(x)
        if self.known_minima is None:
            raise ValueError("No known minima for this function")
        else:
            return min([torch.norm(x - minx, dim=-1) for minx in self.known_minima])

    def plot3D(
        self,
        x_range: torch.Tensor,
        y_range: torch.Tensor,
        sample_path: torch.Tensor = None,
        cmap: cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True),
        path_style: str = "r.-",
    ) -> None:
        X, Y = torch.meshgrid(x_range, y_range)
        Z = torch.tensor(self(torch.stack([X, Y], dim=-1).view(-1, 2))).view(
            len(x_range), len(y_range)
        )

        # plot surface
        offset = 0.5 * (Z.max() - Z.min())
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # plot surface
        ax.contour(X, Y, Z, 50, cmap=cmap, linestyles="solid", offset=-offset)
        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap, linewidth=0.5, antialiased=False, alpha=0.8
        )
        ax.contour(X, Y, Z, 50, colors="k", linestyles="solid")
        ax.set_zlim(-offset, Z.max())

        # plot sample path
        if sample_path is not None:
            ax.plot(
                sample_path[:, 0].numpy(),
                sample_path[:, 1].numpy(),
                sample_path[:, 2].numpy(),
                path_style,
            )

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def set_domain(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        lower = self._check_and_maybe_unsqueeze(lower)
        upper = self._check_and_maybe_unsqueeze(upper)
        self.domain = Domain(lower, upper)

    def out_of_domain(self, x: torch.Tensor) -> bool:
        if self.domain is None:
            return False
        else:
            return not self.domain(x)

    def input_dim(self) -> int:
        return self.layout_info._get_plot_info(self.input_layout)["input_dim"]


# Beale function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def bale_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


class BealeFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(
            lower=torch.tensor([-4.5, -4.5]), upper=torch.tensor([4.5, 4.5])
        )

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return bale_function(vec)


# Goldstein-Price function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def goldstein_price_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        1
        + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + ((2 * x - 3 * y) ** 2)
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


class GoldsteinPriceFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-2, -2]), upper=torch.tensor([2, 2]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return goldstein_price_function(vec)


# Rosenbrock function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def rosenbrock2d_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


class Rosenbrock2DFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-5, -5]), upper=torch.tensor([5, 5]))

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        return rosenbrock2d_function(vec)


# Ackley function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def ackley_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2)))
        - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


class AckleyFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-5, -5]), upper=torch.tensor([5, 5]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return ackley_function(vec)


# hölder table function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def holder_table_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return -torch.abs(
        torch.sin(x)
        * torch.cos(y)
        * torch.exp(torch.abs(1 - torch.sqrt(x**2 + y**2) / np.pi))
    )


class HolderTableFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-10, -10]), upper=torch.tensor([10, 10]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return holder_table_function(vec)


# Himmelblau function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def himmelblau_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


class HimmelblauFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-5, -5]), upper=torch.tensor([5, 5]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return himmelblau_function(vec)


# three hump camel function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def three_hump_camel_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2


class ThreeHumpCamelFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-5, -5]), upper=torch.tensor([5, 5]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return three_hump_camel_function(vec)


# Easom function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def easom_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        -torch.cos(x) * torch.cos(y) * torch.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    )


class EasomFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(
            lower=torch.tensor([-100, -100]), upper=torch.tensor([100, 100])
        )

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return easom_function(vec)


# cross in tray function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def cross_in_tray_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        -0.0001
        * torch.abs(
            torch.sin(x)
            * torch.sin(y)
            * torch.exp(torch.abs(100 - torch.sqrt(x**2 + y**2) / np.pi))
            + 1
        )
        ** 0.1
    )


class CrossInTrayFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-10, -10]), upper=torch.tensor([10, 10]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return cross_in_tray_function(vec)


# McCormick function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def mccormick_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


class McCormickFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-1.5, -3]), upper=torch.tensor([4, 4]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return mccormick_function(vec)


# Bukin function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


def bukin_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return 100 * torch.sqrt(torch.abs(y - 0.01 * x**2)) + 0.01 * torch.abs(x + 10)


class BukinFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(lower=torch.tensor([-15, -5]), upper=torch.tensor([-5, 3]))

    def handele_out_of_domain(self, x: torch.Tensor) -> torch.Tensor:
        # primitive handling of out of domain values
        # project back
        x = self.domain.project(x)

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        vec = self._check_and_maybe_unsqueeze(vec)
        self.handele_out_of_domain(vec)
        return bukin_function(vec)


if __name__ == "__main__":
    print(bale_function(torch.tensor([[1, 2], [2, 2]])))

    beale = BealeFunction()
    print(beale(torch.tensor(torch.tensor([[1, 2], [2, 2]]))))
