"""Improved Block QR-CS explainer with adaptive sparsity detection."""

import numpy as np
from typing import Optional, Callable, Dict, List
from .block_qrcs import BlockQRCSExplainer


class ImprovedBlockQRCSExplainer(BlockQRCSExplainer):
    """Block QR-CS explainer with adaptive sparsity detection per block.

    This extends BlockQRCSExplainer to automatically detect if each block
    is sparse enough for compressed sensing and adapts the method accordingly.

    Key improvements:
    - Per-block sparsity detection
    - Adaptive method selection for each block
    - Can mix CS and fast fallback across blocks
    - Reports sparsity statistics for all blocks
    """

    def __init__(
        self,
        characteristic_function=None,
        n_blocks: int = 4,
        block_sizes: Optional[List[int]] = None,
        n_measurements_per_block: Optional[List[int]] = None,
        tolerance: float = 5e-5,
        use_fast_fallback: Optional[bool] = None,
        sparsity_threshold: float = 0.8,
        auto_adapt: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize improved Block QR-CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            n_blocks: Number of blocks to divide the problem into
            block_sizes: Size of each block (auto-computed if None)
            n_measurements_per_block: Measurements per block (auto-computed if None)
            tolerance: L1 optimization tolerance
            use_fast_fallback: Force fast fallback globally (default keeps legacy behaviour)
            sparsity_threshold: Minimum sparsity ratio to use CS (default 0.8 = 80% sparse)
            auto_adapt: Automatically switch to fast fallback per block if not sparse
            parallel: If True, compute blocks in parallel
            max_workers: Maximum number of parallel workers
            verbose: Whether to print progress
        """
        # Note: we start with use_fast_fallback=False to allow per-block decisions
        super().__init__(
            characteristic_function=characteristic_function,
            n_blocks=n_blocks,
            block_sizes=block_sizes,
            n_measurements_per_block=n_measurements_per_block,
            tolerance=tolerance,
            use_fast_fallback=use_fast_fallback if use_fast_fallback is not None else False,
            parallel=parallel,
            max_workers=max_workers,
            verbose=verbose
        )
        self.sparsity_threshold = sparsity_threshold
        self.auto_adapt = auto_adapt
        self._block_sparsities = {}
        self._block_methods = {}

    def _check_block_sparsity(
        self,
        block_idx: int,
        utility_func: Callable,
        sample_size: int = 30
    ) -> float:
        """Check if marginal contributions in a block are sparse.

        Args:
            block_idx: Index of the block
            utility_func: Utility function for coalition values
            sample_size: Number of coalitions to sample

        Returns:
            Sparsity ratio (fraction of near-zero coefficients)
        """
        start, end = self.block_ranges[block_idx]
        block_size = self.block_sizes[block_idx]
        sample_size = min(sample_size, block_size)

        # Sample marginal contributions for first player in this block
        player = 0
        sample_indices = np.random.choice(
            range(start, min(end, start + sample_size)),
            size=sample_size,
            replace=False
        )

        marginal_contribs = []
        for global_idx in sample_indices:
            coalition = self._index_to_coalition(global_idx, player)
            node_set = {self.nodes[i] for i in coalition}
            node_set_with = node_set | {self.nodes[player]}

            v_with = self.characteristic_function(node_set_with, self.graph)
            v_without = self.characteristic_function(node_set, self.graph) if node_set else 0
            marginal_contribs.append(v_with - v_without)

        if len(marginal_contribs) == 0:
            return 0.0

        u_sample = np.array(marginal_contribs)

        # Create small DCT basis for sample
        Psi_sample = self._get_block_dct_basis(sample_size)

        # Transform to DCT domain
        s_sample = Psi_sample.T @ u_sample

        # Count near-zero coefficients
        threshold = 0.01 * np.max(np.abs(s_sample)) if len(s_sample) > 0 else 0.01
        sparsity = np.sum(np.abs(s_sample) < threshold) / len(s_sample)

        return sparsity

    def _compute_single_block(
        self,
        block_idx: int,
        utility_func: Callable
    ) -> Dict[int, float]:
        """Compute Shapley values for a single block with adaptive method.

        Checks sparsity first and adapts method accordingly for this block.
        """
        # Check sparsity for this block if auto_adapt is enabled
        use_fast_for_block = self.use_fast_fallback  # Default from parent

        if self.auto_adapt and block_idx not in self._block_sparsities:
            sparsity = self._check_block_sparsity(block_idx, utility_func)
            self._block_sparsities[block_idx] = sparsity

            if sparsity < self.sparsity_threshold:
                use_fast_for_block = True
                self._block_methods[block_idx] = 'Fast Fallback'
                if self.verbose:
                    print(f"  Block {block_idx}: Sparsity {100*sparsity:.1f}% < {100*self.sparsity_threshold:.1f}%, using fast fallback")
            else:
                use_fast_for_block = False
                self._block_methods[block_idx] = 'Compressed Sensing'
                if self.verbose:
                    print(f"  Block {block_idx}: Sparsity {100*sparsity:.1f}% >= {100*self.sparsity_threshold:.1f}%, using CS")
        elif block_idx in self._block_sparsities:
            # Use cached decision
            use_fast_for_block = (self._block_methods[block_idx] == 'Fast Fallback')

        # Delegate to parent with explicit override to avoid shared-state races
        return super()._compute_single_block(
            block_idx,
            utility_func,
            use_fast_override=use_fast_for_block
        )

    def get_block_sparsity_info(self) -> Dict[str, any]:
        """Get sparsity information for all blocks.

        Returns:
            Dict with sparsity statistics
        """
        if not self._block_sparsities:
            return {
                'message': 'No sparsity detection performed yet',
                'auto_adapt': self.auto_adapt
            }

        sparsities = list(self._block_sparsities.values())
        methods = list(self._block_methods.values())

        return {
            'n_blocks': self.n_blocks,
            'block_sparsities': self._block_sparsities,
            'block_methods': self._block_methods,
            'avg_sparsity': np.mean(sparsities) if sparsities else 0,
            'min_sparsity': np.min(sparsities) if sparsities else 0,
            'max_sparsity': np.max(sparsities) if sparsities else 0,
            'threshold': self.sparsity_threshold,
            'blocks_using_cs': sum(1 for m in methods if m == 'Compressed Sensing'),
            'blocks_using_fast': sum(1 for m in methods if m == 'Fast Fallback')
        }

    def _compute_shapley_blocks(self, utility_func: Callable) -> Dict[int, float]:
        """Override to add summary statistics after computation."""
        result = super()._compute_shapley_blocks(utility_func)

        if self.verbose and self.auto_adapt:
            info = self.get_block_sparsity_info()
            if 'avg_sparsity' in info:
                print(f"\nBlock sparsity summary:")
                print(f"  Average sparsity: {info['avg_sparsity']*100:.1f}%")
                print(f"  Blocks using CS: {info['blocks_using_cs']}/{self.n_blocks}")
                print(f"  Blocks using fast fallback: {info['blocks_using_fast']}/{self.n_blocks}")

        return result
