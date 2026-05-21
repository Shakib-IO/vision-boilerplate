### PositionEmbeddingSine for 2D inputs

```bash
class PositionEmbeddingSine(nn.Module):
    """A module for generating sinusoidal positional embeddings for 2D inputs like images.

    This class implements sinusoidal position encoding for 2D spatial positions, which can be used in transformer-based
    models for computer vision tasks.

    Attributes:
        num_pos_feats (int): Number of positional features (half of the embedding dimension).
        temperature (int): Temperature parameter for the sinusoidal functions.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): Scaling factor for the embeddings when normalize is True.
        cache (dict): Cache for storing precomputed embeddings.

    Methods:
        _encode_xy: Encodes 2D positions using sine and cosine functions.
        encode_boxes: Encodes box coordinates and dimensions into positional embeddings.
        encode_points: Encodes 2D point coordinates with sinusoidal positional embeddings.
        forward: Generates sinusoidal position embeddings for 2D inputs.

    Examples:
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        """Initialize sinusoidal position embeddings for 2D image inputs."""
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode 2D positions using sine/cosine functions for transformer positional embeddings."""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Encode box coordinates and dimensions into positional embeddings for detection."""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Encode 2D points with sinusoidal embeddings and append labels."""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tensor:
        """Generate sinusoidal position embeddings for 2D inputs like images."""
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

```

### Positional encoding using random spatial frequencies.

```bash
class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies.

    This class generates positional embeddings for input coordinates using random spatial frequencies. It is
    particularly useful for transformer-based models that require position information.

    Attributes:
        positional_encoding_gaussian_matrix (torch.Tensor): A buffer containing random values for encoding.

    Methods:
        _pe_encoding: Positionally encodes points that are normalized to [0,1].
        forward: Generates positional encoding for a grid of the specified size.
        forward_with_coords: Positionally encodes points that are not normalized to [0,1].

    Examples:
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        """Initialize random spatial frequency position embedding for transformers."""
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

        # Set non-deterministic for forward() error 'cumsum_cuda_kernel does not have a deterministic implementation'
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode normalized [0,1] coordinates using random spatial frequencies."""
        # Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # Outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid using random spatial frequencies."""
        h, w = size
        grid = torch.ones(
            (h, w),
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=self.positional_encoding_gaussian_matrix.dtype,
        )
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Positionally encode input coordinates, normalizing them to [0,1] based on the given image size."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C
```
