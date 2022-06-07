---
layout: post
title: "functorch and jax"
author: Guanlin Li
tag: notes
---

> There's a while in last year (2022) that many researchers were talking about `jax` on twitter. `jax` is an open-source project from Google brain which some people have saide it may take over `tensorflow` to be a competitive and flexible library for deep learning like `pytorch`.
>
> As far as I know, `jax` is a `python` package you can `import` for high-order differentiation with naive `numpy`, and may be the future software ground of deep learning. So to get familiar with it and its corresponding `pytorch` counterpart `functorch`, I started this note today (6/2/2022).
>
> There is also content about `autodiff`, since it is the foundation of `jax` and `functorch`.



- toc
{:toc}
## `functorch`


The preceder of `jax` is `autograd` developed mainly by [Dougal Maclaurin](https://dougalmaclaurin.com/) and its collegues during his PhD in Harvard, and then he joined Google brain and developed `jax`.

- `functorch` [a whirlwind tour](https://pytorch.org/functorch/stable/notebooks/whirlwind_tour.html): is a library for `jax`-like composable function transforms in `pytorch`.
  - `functorch` has
    - auto-diff transform `grad(f)` returning a function that computes gradient of `f`
    - a vectorization/batching transform `vmap(f)` return a function that computes `f` over batches of inputs
    - a compilation transform `functorch.compile` namespace, AOT (ahead-of-time) autograd returns an FX graph, so that compilation with different backends can be much easier (?)
  - The reasons to develop `functorch` beyond `pytorch`:
    - Computing per-sample-gradients or other per-sample-quantities
    - Running ensemble of model on a single machine (?)
    - Efficiently batching together tasks in the inner-loop of MAML
    - Efficiently computing (batched) Jacobian/Hessian

Some examples of the transforms in `functorch`.

- `grad`

```python
from functorch import grad
x = torch.randn([])  # a scalar
cos_x = grad(lambda x: torch.sin(x))(x)
assert torch.allclose(cos_x, x.cos())

# Second-order gradients
neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
assert torch.allclose(neg_sin_x, -x.sin())
```

- `vmap` is a transform that adds a dimension to all Tensor operations in `func`. `vmap(func)` returns a new function that can take batches of examples with `vmap(func)`(batch_here), leading to a simpler modeling experience.

```python
import torch
from functorch import vmap
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

# this model definition can hide away the 0-th dim, the batch dimension
def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = vmap(model)(examples)
```

- Combining `grad` and `vmap`

```python
from functorch import vmap
batch_size, feature_size = 3, 5

def model(weights,feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss

weights = torch.randn(feature_size, requires_grad=True)
examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)
inputs = (weights, examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

> The last line gives a weird usage of `functorch`; the `in_dims` arg seems to specify the added dimension of `inputs`.
>
> Wonder how does it look like in `jax`.

- `vjp` (vector-jacobian-product)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)  # this guarantees initialization to be the same for different runs


# the functional form of a neural network
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()


# vector-jacobian product, aka vjp
# the usage of torch.autograd.grad(f, x, v)
# the res is 'v' dot 'grad(f; x)'
# f: R^d, x: R^m, v: R^d
# 1 x d dot d x m -- 1 x m
def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)


D = 4
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D) # feature vector

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jac = compute_jac(xp)
print(jac.shape)
print(jac[0])
```

- `vjp` + `vmap`: instead of using for-loop to compute Jacobian row-by-row, use `vmap` to vectorise the computati

  - > The `vjp` doc is [here](https://pytorch.org/functorch/stable/generated/functorch.vjp.html#functorch.vjp).
    >
    > - Returns a tuple (output, vjp_fn)

```python
# currently 6/3/2022, I can't understand the usage here
from functorch import vmap, vjp

# the usage of vjp
_, vjp_fn = vjp(
  partial(predict, weight, bias),  # a python function that takes one/more args, must return one/more tensors
  x  # primals: positional args to func that must all be tensors, over which derivatives are computed from
)

# the usage of vmap
ft_jacobian, = vmap(vjp_fn)(unit_vectors)

# lets confirm both methods compute the same result
assert torch.allclose(ft_jacobian, jac)
```

- `jvp` (Jacobian-vector-product)



## `jax`



### `jax` ecosystem





## Reference

- [CSC321 Lecture 10: Automatic Differentiation](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf) by Roger Grosse.

> - What autodiff is?
>
>   - An autodiff system should transform the left-hand side into the right-hand sid
>
>     - Left: computing loss
>       - $$z = wx + b, y = \sigma(z), \mathcal{L} = \frac{1}{2} (y - t)^2$$
>     - Right: computing derivatives
>       - $$\bar{\mathcal{L}} = 1, \bar{y}=y - t, \bar{z} = \bar{y} \sigma'(z), \bar{w}=\bar{z}x, \bar{b}=\bar{z}$$
>
>   - An autodiff system will convert the program into a sequence of primitive operations which have specified routines for computing derivatives.
>
>   - **Building the Computation Graph**
>
>     - ```python
>       # with primitives
>       
>       def logistic(z):
>         return 1. / (1. + np.exp(-z))
>       
>       # that is equivalent to:
>       def logistic_prim(z):
>         return np.reciprocal(
>         	np.add(
>           	1,
>             np.exp(
>             	np.negative(z)
>             )
>           )
>         )
>       ```
>
>     - **Vector-Jacobian Products**
>
>       - The backdrop equation (single child node) can be written as a vector-Jacobin product (VJP):
>       - $$\bar{x_j} = \sum_i \bar{y_i} \frac{\partial y_i}{\partial x_j} $$ $$\rightarrow$$ $$\bar{x} = \bar{y}^T J$$, where $$J$$ is the Jacobian matrix ($$m \times n$$)
>       - That gives a row vector ($$1 \times n$$), so we can treat it as a column vextor by taking: $$\bar{x} = J^T \bar{y}$$
>         - Matrix-vector product
>         - Element-wise op
>       - **Note**: we never explicitly construct a Jacobian, it's usually simpler and more efficient to compute the VJP directly.
>
>   - Backward Pass
>
>     - View backdrop as message passing

- [Optional Reading: Vector Calculus using autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#optional-reading-vector-calculus-using-autograd) on pytorch autograd tutorial.

  - > Cite a few sentences here: 
    >
    > *Generally speaking, `torch.autograd` is an engine for computing vector-Jacobian product. That is given any vector $$v$$, compute $$J^T \cdot v$$*
    >
    > *`external_grad` represents $$v$$*

- [fmin.xyz](https://fmin.xyz/), a very concise but systematic review of optimisation theory.

