import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from utils import maybe_unsqueeze


class InputLayout:
    input_layouts = [
        "X", "BX", "XY", "BXY", "XYZ", "BXYZ"
    ]
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
        space_dim = dim - 1 if multisamples else dim
        last_dim_shape = self[input_layout + "_SHAPE"]
        return {
            "total_dim": dim, 
            "space_dim": space_dim, 
            "multisamples": multisamples,
            "last_dim_shape": last_dim_shape
            }

@dataclass
class Domain:
    lower: torch.Tensor
    upper: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return all(torch.le(self.lower, x).view(-1)) and all(torch.le(x, self.upper).view(-1))


class TestFunctionBase(ABC):
    layout_info: InputLayout = InputLayout()

    def __init__(self, input_layout: str):
        self.input_layout: str = input_layout
        self.input_layout_dim: int = self.layout_info[input_layout]
        self.domain: Domain = None

    @abstractmethod
    def __call__(self, vec: torch.Tensor) -> torch.Tensor:
        pass

    def _check_and_maybe_unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > self.input_layout_dim:
            raise ValueError(
                f"Input tensor has {x.dim()} dimensions, but expected dim <= {self.input_layout_dim}"
            )
        elif x.dim() < self.input_layout_dim - 1 and self.layout_info.has_batch_dim(self.input_layout):
            raise ValueError(
                f"Input tensor with dimension {x.dim()} is underdefined for layout {self.input_layout}"
            )
        elif x.dim() < self.input_layout_dim and not self.layout_info.has_batch_dim(self.input_layout):
            raise ValueError(
                f"Input tensor with dimension {x.dim()} is underdefined for layout {self.input_layout}"
            )

        x = maybe_unsqueeze(x, self.input_layout_dim)
        return x
          
    def plot(self, vec: torch.Tensor, **kwargs) -> None:
        # TODO: implement
        raise NotImplementedError

    def set_domain(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        lower = self._check_and_maybe_unsqueeze(lower)
        upper = self._check_and_maybe_unsqueeze(upper)
        self.domain = Domain(lower, upper)

    def out_of_domain(self, x: torch.Tensor) -> bool:
        if self.domain is None:
            return False
        else:
            return not self.domain(x)




# Beale function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

def bale_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

class BealeFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(torch.tensor([-4.5, -4.5]), torch.tensor([4.5, 4.5]))

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        return bale_function(vec)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_and_maybe_unsqueeze(x)
        if self.out_of_domain(x):
            raise ValueError("Input is out of domain")
        return self.f(x)


# Goldstein-Price function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

def goldstein_price_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (
        1
        + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
        + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    )

class GoldsteinPriceFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")
        self.set_domain(torch.tensor([-2, -2]), torch.tensor([2, 2]))

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        return goldstein_price_function(vec)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_and_maybe_unsqueeze(x)
        if self.out_of_domain(x):
            raise ValueError("Input is out of domain")
        return self.f(x)


# Rosenbrock function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

def rosenbrock2d_function(vec: torch.Tensor) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

class Rosenbrock2DFunction(TestFunctionBase):
    def __init__(self) -> None:
        super().__init__(input_layout="BXY")

    def f(self, vec: torch.Tensor) -> torch.Tensor:
        return rosenbrock2d_function(vec)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_and_maybe_unsqueeze(x)
        return self.f(x)


        



if __name__ == "__main__":
    print(bale_function(torch.tensor([[1, 2],[2,2]])))

    beale = BealeFunction()
    print(beale(torch.tensor(torch.tensor([[1, 2],[2,2]]))))