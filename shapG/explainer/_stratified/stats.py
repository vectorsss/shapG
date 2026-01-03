"""Statistics data class for stratified sampling."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SamplingStats:
    """Statistics about the stratified sampling process."""

    n_players: int
    total_budget: int
    allocations_by_size: np.ndarray
    actual_samples_by_size: np.ndarray
    coverage_by_size: np.ndarray  # fraction of coalitions sampled per stratum

    def __str__(self) -> str:
        lines = [f"SamplingStats(n={self.n_players}, budget={self.total_budget})"]
        for s in range(len(self.allocations_by_size)):
            if self.allocations_by_size[s] > 0:
                lines.append(
                    f"  size {s}: {int(self.actual_samples_by_size[s])} samples "
                    f"({self.coverage_by_size[s]*100:.1f}% coverage)"
                )
        return "\n".join(lines)
