from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, runtime_checkable

from typing_extensions import Protocol

# from .scalar import Scalar
# from .scalar import Scalar
from .operators import zipWith

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    arr_plus = list(vals)
    arr_minus = list(vals)

    arr_plus[arg] += epsilon
    arr_minus[arg] -= epsilon

    f_plus = f(*arr_plus)
    f_minus = f(*arr_minus)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


@runtime_checkable
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    def visit(v: Variable):
        if v in visited:
            return
        visited.add(v)
        if v.history is not None:
            for node in v.history.inputs:
                if node not in visited:
                    visit(node)
        L.append(v)

    L = []
    visited = set()
    visit(variable)
    return L


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    print(f"{variable=}")
    print(f"{deriv=}")
    topo_order = topological_sort(variable)
    grads: dict[int, float] = {variable.unique_id: deriv}
    for v in reversed(topo_order):
        d_output = grads.get(v.unique_id, 0.0)
        """print(f'{d_output=}')
        print(f'{v.history=}')
        print(f'{v=}')"""
        if v.history.last_fn is None:
            v.accumulate_derivative(d_output)
            # print(f'{v=}, {d_output=}')
        else:
            for parent, d_input in v.chain_rule(d_output):
                grads[parent.unique_id] = grads.get(parent.unique_id, 0.0) + d_input
        # print(grads)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
