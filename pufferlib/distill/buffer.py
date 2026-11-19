"""FIFO buffer of teacher labels for online BC and DAgger."""

import torch

from pufferlib.distill.bicycle import NU


class LabelBuffer:
    """Ring buffer of (obs, label) pairs on one device. Obs are stored in
    fp16 to fit ~1M samples; labels stay fp32."""

    def __init__(
            self,
            capacity: int,
            obs_dim: int,
            device: str,
            seed: int = 0,
        ) -> None:
        """
        Allocate the storage up front.

        Args:
            capacity (int): maximum number of stored samples.
            obs_dim (int): observation width.
            device (str): torch device for storage and sampling.
            seed (int): seed of the sampling generator.
        """
        if capacity <= 0 or obs_dim <= 0:
            raise ValueError("capacity and obs_dim must be positive")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        self.obs = torch.zeros(
            capacity, obs_dim, dtype=torch.float16, device=device
        )
        self.labels = torch.zeros(
            capacity, NU, dtype=torch.float32, device=device
        )
        self.next_row = 0
        self.filled_count = 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def add(
            self,
            obs: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
        """
        Append rows, overwriting the oldest once full.

        Args:
            obs (torch.Tensor): (n,obs_dim) observations.
            labels (torch.Tensor): (n,2) teacher controls.
        """
        row_count = obs.shape[0]
        if obs.shape != (row_count, self.obs_dim):
            raise ValueError(f"obs shape {tuple(obs.shape)}")
        if labels.shape != (row_count, NU):
            raise ValueError(f"labels shape {tuple(labels.shape)}")
        if row_count > self.capacity:
            raise ValueError("more rows than the buffer holds")
        if row_count == 0:
            return
        if not (torch.isfinite(obs).all() and torch.isfinite(labels).all()):
            raise ValueError("non-finite obs or labels")
        # Rows wrap around the end of the storage
        rows = (
            self.next_row
            + torch.arange(row_count, device=self.device)
        ) % self.capacity
        self.obs[rows] = obs.to(self.device, torch.float16)
        self.labels[rows] = labels.to(self.device, torch.float32)
        self.next_row = (self.next_row + row_count) % self.capacity
        self.filled_count = min(self.filled_count + row_count, self.capacity)

    def sample(
            self,
            batch_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uniform sample with replacement over the filled rows.

        Args:
            batch_size (int): number of rows.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: obs (batch_size,obs_dim)
                fp32 and labels (batch_size,2).
        """
        if self.filled_count == 0:
            raise ValueError("sample() on an empty buffer")
        rows = torch.randint(
            self.filled_count,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return self.obs[rows].float(), self.labels[rows]

    def snapshot(
            self,
            count: int,
            window: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evenly spaced rows over the newest `window` rows, oldest first.
        Draws no random numbers, so it leaves sample() untouched.

        Args:
            count (int): rows wanted; fewer if the window is smaller.
            window (int | None): newest rows to span; None spans every
                filled row.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: obs (n,obs_dim) fp32 and
                labels (n,2), copies.
        """
        if self.filled_count == 0:
            raise ValueError("snapshot() on an empty buffer")
        span = self.filled_count
        if window is not None:
            span = min(window, self.filled_count)
        row_count = min(count, span)
        offsets = torch.linspace(
            0, span - 1, row_count, device=self.device
        ).round().long()
        rows = (self.next_row - span + offsets) % self.capacity
        return self.obs[rows].float(), self.labels[rows].clone()
