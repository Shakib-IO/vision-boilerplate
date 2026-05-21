### Drop Path

DropPath forces the model to:
* not rely too much on specific blocks.
* learn more robust representations.
* behave like an ensemble of shallower/deeper networks.
It improves generalization.

```bash
class DropPath(nn.Module):
    """Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to the input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath module for stochastic depth regularization during training."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth to input tensor during training, with optional scaling."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
```
