"""
Lab 5 - Tasks 1-3: a scalar reverse-mode automatic differentiation engine.

Every arithmetic operation on Value objects creates a new Value node that
remembers (a) its numeric result, (b) its parent nodes and (c) a closure that
applies the LOCAL derivative rule (chain rule for one node).  Calling
.backward() on the output topologically sorts the graph and runs those
closures in reverse order, ACCUMULATING gradients with +=.
"""
import math


class Value:
    """One node of the computation graph (Task 1).

    Attributes
    ----------
    data      : float   - value computed in the forward pass
    grad      : float   - d(output)/d(this node), filled by backward()
    _prev     : tuple   - parent nodes this node was computed from
    _op       : str     - name of the operation (for printing / debugging)
    _backward : callable- local derivative rule: pushes self.grad to parents
    """

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None   # leaf nodes have no rule
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    # ------------------------------------------------------------------ #
    # Task 2: core operations. Each one builds a graph node in forward.  #
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                       # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                       # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                       # d(x^k)/dx = k x^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():                       # 1 if x>0 else 0 (PyTorch convention at 0)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # extra primitives needed for Task 5 (sigmoid + BCE loss)
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():                       # d(e^x)/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                       # d(ln x)/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():                       # s(1-s)
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    # convenience operators built from the primitives above
    def __neg__(self):            return self * -1
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return other + (-self)
    def __radd__(self, other):    return self + other
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    # ------------------------------------------------------------------ #
    # Task 3: backward pass                                              #
    # ------------------------------------------------------------------ #
    def backward(self):
        # 1) topological sort (iterative DFS -> no recursion limit on big graphs)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)              # all parents already appended
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        # 2) seed: d(out)/d(out) = 1
        self.grad = 1.0
        # 3) visit nodes in REVERSE topological order; each _backward uses +=
        for node in reversed(topo):
            node._backward()
        return topo


def vsum(values):
    """sum() for a list of Values (the .sum() in Task 4)."""
    out = Value(0.0)
    for v in values:
        out = out + v
    return out


def zero_grad(params):
    for p in params:
        p.grad = 0.0
