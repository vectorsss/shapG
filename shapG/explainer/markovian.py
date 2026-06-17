"""
Markovian greedy Shapley-Weber process explainer.

Implements a greedy coalition formation process based on the framework of
"Values for Markovian coalition processes" (Faigle & Grabisch, 2012).

The algorithm builds a coalition greedily from the empty set, adding the player
with the maximum marginal contribution at each step. Since exactly one player
changes per step, Shapley I = Shapley II = the player's marginal contribution.

Complexity: O(n^2) characteristic function evaluations.
"""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from ..characteristic.characteristic_functions import CoalitionDegree
from .base import CharacteristicFunction, Explainer

logger = logging.getLogger(__name__)

__all__ = ["MarkovianExplainer"]


class MarkovianExplainer(Explainer):
    """Greedy Shapley-Weber process explainer.

    Builds a coalition greedily from the empty set, adding the player with the
    maximum marginal contribution at each step. Stops when no addition improves
    the characteristic function value. Each player's Shapley value equals their
    marginal contribution when they were added (or 0 if never added).

    When ``allow_leave=True``, the process also considers removing players from
    the coalition, enabling escape from local optima. The last player added
    cannot be immediately removed to prevent trivial oscillation.

    Parameters
    ----------
    characteristic_function : CharacteristicFunction, optional
        Function to compute coalition values. Defaults to CoalitionDegree().
    verbose : bool, default=False
        Whether to print progress information.
    seed : int, optional
        Random seed for tie-breaking when multiple players have equal
        marginal contributions.
    allow_leave : bool, default=False
        If True, enable the join-and-leave process where players can leave
        the coalition when doing so improves the characteristic function value.
    max_iterations : int, optional
        Maximum number of steps for the process. Only used when
        ``allow_leave=True``. Defaults to ``3 * n`` if not specified.
    initial_coalition : list, optional
        Pre-specified starting coalition (external player IDs). The greedy
        process starts from this set instead of the empty set. Players in
        the initial coalition receive Shapley values equal to their marginal
        contribution in the context of the initial set.
    min_improvement : float, default=0.0
        Minimum improvement threshold for accepting join/leave actions.
        Positive values (e.g., 0.001) make the process stricter, only
        accepting clearly beneficial moves. Negative values (e.g., -0.001)
        make it more lenient, tolerating slightly harmful moves.

    Examples
    --------
    >>> from shapG import MarkovianExplainer, CoalitionDegree
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.3, seed=42)
    >>> explainer = MarkovianExplainer(seed=42)
    >>> values = explainer.fit_explain(G)

    Join-and-leave mode:

    >>> explainer = MarkovianExplainer(seed=42, allow_leave=True)
    >>> values = explainer.fit_explain(G)
    >>> history = explainer.get_process_history()

    With initial coalition and threshold:

    >>> explainer = MarkovianExplainer(
    ...     seed=42, initial_coalition=[0, 1, 2], min_improvement=0.001
    ... )
    >>> values = explainer.fit_explain(G)
    """

    # Supported transition strategies
    STRATEGIES = ("greedy", "linear", "softmax", "top_p", "top_k", "min_p")

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        allow_leave: bool = False,
        max_iterations: Optional[int] = None,
        initial_coalition: Optional[List] = None,
        min_improvement: float = 0.0,
        strategy: str = "greedy",
        temperature: float = 1.0,
        strategy_param: Optional[float] = None,
        n_scenarios: int = 1,
    ):
        super().__init__(characteristic_function or CoalitionDegree(), verbose)
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {self.STRATEGIES}"
            )
        self.seed = seed
        self.allow_leave = allow_leave
        self.max_iterations = max_iterations
        self.initial_coalition = initial_coalition
        self.min_improvement = min_improvement
        self.strategy = strategy
        self.temperature = temperature
        self.strategy_param = strategy_param
        self.n_scenarios = n_scenarios
        self._rng = np.random.default_rng(seed)

        # Set during fit
        self.n = None
        self.player_ids = None
        self._context = None

        # Set during explain
        self._shapley_values = None
        self._greedy_order = None
        self._final_coalition = None
        self._marginal_records = None
        self._process_history = None
        self._scenario_finals = None
        self._scenario_orders = None
        self._absorbing_metrics = None  # v(S*) per scenario
        self._full_set_metric = None  # v(N)
        self._transition_counts = None  # {S: {T: count}}
        self._scenario_paths = None  # list of paths per scenario
        self._transition_probs = None  # {S: {T: exact prob at sampling time}}

    def fit(
        self,
        X: Union[int, np.ndarray, pd.DataFrame, nx.Graph, list],
        context: Optional[Any] = None,
        **kwargs,
    ) -> "MarkovianExplainer":
        """Fit the explainer to data.

        Parameters
        ----------
        X : int, array-like, DataFrame, Graph, or List
            Input that defines players. Can be:
            - int: number of players directly
            - nx.Graph: uses nodes as players
            - pd.DataFrame: uses columns as players
            - np.ndarray: uses columns (2D) or elements (1D) as players
            - List: uses elements as player identifiers
        context : Any, optional
            Context passed to characteristic function.
            If None and X is Graph/DataFrame/array, X is used as context.

        Returns
        -------
        self
        """
        self.n, self.player_ids, self._context = self._parse_input(X, context)

        # Reset results from previous runs
        self._shapley_values = None
        self._greedy_order = None
        self._final_coalition = None
        self._marginal_records = None
        self._process_history = None
        self._transition_probs = None

        if self.verbose:
            print(f"MarkovianExplainer fitted: n_players={self.n}")

        self._fitted = True
        return self

    @staticmethod
    def _make_cached_utility(utility_func):
        """Wrap utility_func with LRU cache to avoid redundant evaluations."""
        cache = {}

        def cached(internal_set):
            key = frozenset(internal_set)
            if key not in cache:
                cache[key] = utility_func(internal_set)
            return cache[key]

        cached.cache = cache
        return cached

    def _select_action(self, join_deltas, leave_deltas):
        """Select an action (join or leave) based on the current strategy.

        Args:
            join_deltas: dict {player_index: delta_plus} for positive join candidates.
            leave_deltas: dict {player_index: delta_minus} for positive leave candidates.

        Returns:
            Tuple (action, player_index, improvement, action_probs) where
            ``action_probs`` maps every candidate ``(action, player_index)``
            pair to its selection probability under the current strategy.
            Returns (None, None, None, {}) if no positive candidate exists.
        """
        # Merge all candidates passed from pre-filtering
        candidates = []
        for i, d in join_deltas.items():
            candidates.append(("join", i, d))
        for j, d in leave_deltas.items():
            candidates.append(("leave", j, d))

        if not candidates:
            return None, None, None, {}

        if self.strategy == "greedy":
            best = max(candidates, key=lambda x: x[2])
            # Tie-breaking
            tied = [c for c in candidates if abs(c[2] - best[2]) < 1e-12]
            action_probs = {(a, p): 1.0 / len(tied) for a, p, _ in tied}
            if len(tied) > 1:
                idx = self._rng.integers(len(tied))
                return (*tied[idx], action_probs)
            return (*best, action_probs)

        # For stochastic strategies, prefer strictly positive deltas
        positive = [c for c in candidates if c[2] > 0]
        if positive:
            candidates = positive
        else:
            # No positive deltas; fall back to uniform among non-negative (d>=0)
            non_neg = [c for c in candidates if c[2] >= 0]
            if not non_neg:
                return None, None, None, {}
            action_probs = {(a, p): 1.0 / len(non_neg) for a, p, _ in non_neg}
            idx = self._rng.integers(len(non_neg))
            return (*non_neg[idx], action_probs)

        deltas = np.array([c[2] for c in candidates])
        tau = self.temperature

        # Compute softmax base distribution (used by all stochastic strategies)
        logits = deltas / tau
        logits -= logits.max()  # numerical stability
        exp_d = np.exp(logits)
        softmax_probs = exp_d / exp_d.sum()

        if self.strategy == "linear":
            probs = deltas / deltas.sum()

        elif self.strategy == "softmax":
            probs = softmax_probs

        elif self.strategy == "top_k":
            k = int(self.strategy_param) if self.strategy_param else 1
            k = min(k, len(candidates))
            top_indices = np.argsort(softmax_probs)[-k:]
            probs = np.zeros(len(candidates))
            probs[top_indices] = softmax_probs[top_indices]
            probs /= probs.sum()

        elif self.strategy == "top_p":
            p = self.strategy_param if self.strategy_param is not None else 0.9
            sorted_idx = np.argsort(softmax_probs)[::-1]
            cumsum = np.cumsum(softmax_probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, p) + 1
            cutoff = min(cutoff, len(candidates))
            top_indices = sorted_idx[:cutoff]
            probs = np.zeros(len(candidates))
            probs[top_indices] = softmax_probs[top_indices]
            probs /= probs.sum()

        elif self.strategy == "min_p":
            p = self.strategy_param if self.strategy_param is not None else 0.1
            threshold = p * softmax_probs.max()
            mask = softmax_probs >= threshold
            probs = np.where(mask, softmax_probs, 0.0)
            probs /= probs.sum()

        action_probs = {(a, p): float(pr) for (a, p, _), pr in zip(candidates, probs)}
        idx = self._rng.choice(len(candidates), p=probs)
        return (*candidates[idx], action_probs)

    def explain(
        self,
        X: Optional[Union[int, np.ndarray, pd.DataFrame, nx.Graph, list]] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> Dict:
        """Compute Shapley values via greedy Markovian process.

        Parameters
        ----------
        X : optional
            Input data (uses fitted data if None).
        context : Any, optional
            Context for characteristic function.

        Returns
        -------
        Dict
            Shapley values keyed by player identifier.
        """
        if X is not None:
            self.fit(X, context=context, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        def _raw_utility(internal_set: Set[int]) -> float:
            player_set = {self.player_ids[i] for i in internal_set}
            return self.characteristic_function(player_set, self._context)

        # Wrap with cache to avoid redundant evaluations across scenarios
        utility_func = self._make_cached_utility(_raw_utility)

        # Map initial_coalition (external IDs) to internal indices
        initial_internal = set()
        if self.initial_coalition is not None:
            id_to_internal = {pid: i for i, pid in enumerate(self.player_ids)}
            for pid in self.initial_coalition:
                if pid not in id_to_internal:
                    raise ValueError(
                        f"Initial coalition member '{pid}' not found in players."
                    )
                initial_internal.add(id_to_internal[pid])

        M = self.n_scenarios
        if M > 1 and self.strategy == "greedy":
            logger.warning(
                "n_scenarios > 1 with greedy strategy produces identical paths. "
                "Consider using a stochastic strategy."
            )

        if M == 1:
            # Single scenario (backward compatible)
            if self.allow_leave:
                sv_internal, order, final, records, history = (
                    self._run_greedy_process_with_leave(utility_func, initial_internal)
                )
            else:
                sv_internal, order, final, records = self._run_greedy_process(
                    utility_func, initial_internal
                )
                history = self._build_monotone_history(
                    order, records, utility_func, initial_internal
                )
            self._scenario_finals = None
            self._scenario_orders = None
            self._absorbing_metrics = [utility_func(final)]
            self._full_set_metric = utility_func(set(range(self.n)))
            # Extract transitions from history
            tc = {}
            tp = {}
            init_fset = frozenset(initial_internal) if initial_internal else frozenset()
            non_initial = [e for e in history if e["action"] != "initial"]
            prev = init_fset
            path = [init_fset]
            for entry in non_initial:
                cur = entry["coalition"]
                tc.setdefault(prev, {})
                tc[prev][cur] = tc[prev].get(cur, 0) + 1
                # First-wins: revisits of the same state may differ slightly
                # (leave candidates exclude the last-added player)
                if "transition_probs" in entry and prev not in tp:
                    tp[prev] = entry["transition_probs"]
                path.append(cur)
                prev = cur
            self._transition_counts = tc
            self._transition_probs = tp
            self._scenario_paths = [path]
        else:
            # Multi-scenario: sample M paths and average scenario-values
            all_sv = []
            all_orders = []
            all_finals = []
            all_absorbing_metrics = []
            all_paths = []
            tc = {}  # aggregated transition counts
            tp = {}  # exact transition distributions (first-wins per state)
            init_fset = frozenset(initial_internal) if initial_internal else frozenset()
            last_records = None
            last_history = None
            for m in range(M):
                if self.allow_leave:
                    sv_m, order_m, final_m, rec_m, hist_m = (
                        self._run_greedy_process_with_leave(
                            utility_func, initial_internal
                        )
                    )
                else:
                    sv_m, order_m, final_m, rec_m = self._run_greedy_process(
                        utility_func, initial_internal
                    )
                    hist_m = self._build_monotone_history(
                        order_m, rec_m, utility_func, initial_internal
                    )
                all_sv.append(sv_m)
                all_orders.append(order_m)
                all_finals.append(final_m)
                all_absorbing_metrics.append(utility_func(final_m))
                last_records = rec_m
                last_history = hist_m

                # Extract transitions from this scenario's history
                non_initial = [e for e in hist_m if e["action"] != "initial"]
                prev = init_fset
                path = [init_fset]
                for entry in non_initial:
                    cur = entry["coalition"]
                    tc.setdefault(prev, {})
                    tc[prev][cur] = tc[prev].get(cur, 0) + 1
                    if "transition_probs" in entry and prev not in tp:
                        tp[prev] = entry["transition_probs"]
                    path.append(cur)
                    prev = cur
                all_paths.append(path)

            # Average scenario-values across M paths
            sv_internal = {i: 0.0 for i in range(self.n)}
            for sv_m in all_sv:
                for i, v in sv_m.items():
                    sv_internal[i] += v / M

            # Derive order from averaged values (descending)
            ranked = sorted(sv_internal.items(), key=lambda x: x[1], reverse=True)
            order = [i for i, v in ranked if v > 0]

            # Final coalition = features with positive averaged value
            final = frozenset(order)

            records = last_records
            history = last_history

            # Store per-scenario details for metadata
            self._scenario_finals = [
                frozenset(self.player_ids[i] for i in f) for f in all_finals
            ]
            self._scenario_orders = [
                [self.player_ids[i] for i in o] for o in all_orders
            ]
            self._absorbing_metrics = all_absorbing_metrics
            self._full_set_metric = utility_func(set(range(self.n)))
            self._transition_counts = tc
            self._transition_probs = tp
            self._scenario_paths = all_paths

            if self.verbose:
                n_cached = len(utility_func.cache)
                unique_finals = len(set(all_finals))
                sizes = [len(f) for f in all_finals]
                print(
                    f"  {M} scenarios completed. "
                    f"{unique_finals} unique absorbing states "
                    f"(sizes: {min(sizes)}-{max(sizes)}). "
                    f"{n_cached} unique coalitions cached."
                )

        # Map internal indices back to player ids
        self._shapley_values = {self.player_ids[i]: v for i, v in sv_internal.items()}
        self._greedy_order = [self.player_ids[i] for i in order]
        self._final_coalition = frozenset(self.player_ids[i] for i in final)
        self._marginal_records = records
        self._process_history = history

        return self._shapley_values

    def _run_greedy_process(
        self,
        utility_func,
        initial_set: Optional[Set[int]] = None,
    ) -> Tuple[Dict[int, float], List[int], frozenset, List[dict]]:
        """Run the greedy Shapley-Weber coalition formation process.

        Args:
            utility_func: Maps internal-index sets to coalition values.
            initial_set: Internal indices of pre-specified starting coalition.

        Returns:
            Tuple of (shapley_values, greedy_order, final_coalition, marginal_records)
            where marginal_records stores all evaluated marginals for interaction graph.
        """
        S = set(initial_set) if initial_set else set()
        v_S = utility_func(S)
        shapley_values = {i: 0.0 for i in range(self.n)}
        greedy_order = []
        remaining = set(range(self.n)) - S
        marginal_records = []

        # Compute Shapley values for initial coalition members
        if initial_set:
            initial_marginals = {}
            for i in sorted(initial_set):
                mc = v_S - utility_func(S - {i})
                shapley_values[i] = mc
                initial_marginals[i] = mc
            # Add initial players to greedy_order sorted by marginal (desc)
            sorted_initial = sorted(
                initial_set, key=lambda x: initial_marginals[x], reverse=True
            )
            greedy_order.extend(sorted_initial)

            if self.verbose:
                names = [self.player_ids[i] for i in sorted_initial]
                print(f"  Initial coalition: {names} (v={v_S:.6f})")
                for i in sorted_initial:
                    print(
                        f"    {self.player_ids[i]}: "
                        f"marginal={initial_marginals[i]:.6f}"
                    )

        for step in range(self.n):
            if not remaining:
                break

            # Evaluate marginal contribution of every candidate
            best_i = None
            best_marginal = -np.inf
            candidates = sorted(remaining)
            marginals = {}

            for i in candidates:
                v_Si = utility_func(S | {i})
                mc = v_Si - v_S
                marginals[i] = mc

                # Record for interaction graph computation
                marginal_records.append(
                    {
                        "player": i,
                        "coalition": frozenset(S),
                        "marginal": mc,
                    }
                )

                if mc > best_marginal:
                    best_marginal = mc
                    best_i = i

            # Tie-breaking: if multiple players share max marginal, pick randomly
            tied = [i for i in candidates if marginals[i] == best_marginal]
            if len(tied) > 1:
                best_i = self._rng.choice(tied)

            # Early stopping: best marginal below threshold
            if best_marginal < self.min_improvement:
                if self.verbose:
                    print(
                        f"  Step {step + 1}: stopping (max marginal "
                        f"{best_marginal:.6f} < threshold {self.min_improvement})"
                    )
                break

            # Add the best player
            shapley_values[best_i] = best_marginal
            S.add(best_i)
            v_S = utility_func(S)
            remaining.remove(best_i)
            greedy_order.append(best_i)

            if self.verbose:
                print(
                    f"  Step {step + 1}: added player {self.player_ids[best_i]} "
                    f"(marginal={best_marginal:.6f})"
                )

        final_coalition = frozenset(S)
        return shapley_values, greedy_order, final_coalition, marginal_records

    def _run_greedy_process_with_leave(
        self,
        utility_func,
        initial_set: Optional[Set[int]] = None,
    ) -> Tuple[Dict[int, float], List[int], frozenset, List[dict], List[dict]]:
        """Run the join-and-leave Shapley-Weber coalition formation process.

        Extends the monotone greedy process by allowing players to leave the
        coalition when removal improves the characteristic function value. This
        enables escape from local optima at the cost of a non-monotone process.

        Args:
            utility_func: Maps internal-index sets to coalition values.
            initial_set: Internal indices of pre-specified starting coalition.

        Returns:
            Tuple of (shapley_values, greedy_order, final_coalition,
                       marginal_records, process_history)
        """
        max_iter = (
            self.max_iterations if self.max_iterations is not None else 3 * self.n
        )
        S = set(initial_set) if initial_set else set()
        v_S = utility_func(S)
        greedy_order = []
        marginal_records = []
        process_history = []
        last_added = None
        initial_fset = frozenset(S)
        visited_coalitions = {initial_fset}
        scenario_values = {i: 0.0 for i in range(self.n)}

        # Compute Shapley values for initial coalition members
        if initial_set:
            initial_marginals = {}
            for i in sorted(initial_set):
                mc = v_S - utility_func(S - {i})
                initial_marginals[i] = mc
                scenario_values[i] += mc
            sorted_initial = sorted(
                initial_set, key=lambda x: initial_marginals[x], reverse=True
            )
            greedy_order.extend(sorted_initial)

            # Add initial steps to process history
            for idx, i in enumerate(sorted_initial):
                process_history.append(
                    {
                        "step": idx + 1,
                        "action": "initial",
                        "player": i,
                        "improvement": initial_marginals[i],
                        "coalition": frozenset(S),
                        "coalition_size": len(S),
                        "value": v_S,
                    }
                )

            if self.verbose:
                names = [self.player_ids[i] for i in sorted_initial]
                print(f"  Initial coalition: {names} (v={v_S:.6f})")
                for i in sorted_initial:
                    print(
                        f"    {self.player_ids[i]}: "
                        f"marginal={initial_marginals[i]:.6f}"
                    )

        step_offset = len(process_history)

        for step in range(max_iter):
            step_num = step_offset + step + 1

            # Evaluate JOIN candidates: players not in S
            join_candidates = sorted(set(range(self.n)) - S)
            join_deltas = {}
            for i in join_candidates:
                v_Si = utility_func(S | {i})
                mc = v_Si - v_S
                marginal_records.append(
                    {
                        "player": i,
                        "coalition": frozenset(S),
                        "marginal": mc,
                        "action": "join",
                    }
                )
                join_deltas[i] = mc

            # Evaluate LEAVE candidates: players in S (except last_added)
            leave_candidates = sorted(i for i in S if i != last_added)
            leave_deltas = {}
            for i in leave_candidates:
                v_Si = utility_func(S - {i})
                mc = v_Si - v_S  # positive means removal helps
                marginal_records.append(
                    {
                        "player": i,
                        "coalition": frozenset(S),
                        "marginal": -mc,  # store as marginal of being in S
                        "action": "leave",
                    }
                )
                leave_deltas[i] = mc

            # Filter by min_improvement threshold before selecting
            filtered_join = {
                i: d for i, d in join_deltas.items() if d >= self.min_improvement
            }
            filtered_leave = {
                i: d for i, d in leave_deltas.items() if d > self.min_improvement
            }

            # Select action via strategy
            action, player, improvement, action_probs = self._select_action(
                filtered_join, filtered_leave
            )
            # Exact transition distribution from the *current* S, keyed by
            # target coalition (persisted so plots can label sampled edges
            # with exact probabilities at no extra cost).
            step_transition_probs = {
                (frozenset(S | {p}) if act == "join" else frozenset(S - {p})): float(pr)
                for (act, p), pr in action_probs.items()
            }

            if action is None:
                # No improving move — absorbing state
                if self.verbose:
                    parts = [f"  Step {step_num}: stopping"]
                    best_join = max(join_deltas.values()) if join_deltas else None
                    best_leave = max(leave_deltas.values()) if leave_deltas else None
                    if best_join is not None:
                        bj_i = max(join_deltas, key=join_deltas.get)
                        parts.append(
                            f"best join: {self.player_ids[bj_i]} " f"({best_join:+.6f})"
                        )
                    if best_leave is not None:
                        bl_i = max(leave_deltas, key=leave_deltas.get)
                        parts.append(
                            f"best leave: {self.player_ids[bl_i]} "
                            f"({best_leave:+.6f})"
                        )
                    print(f"{parts[0]} ({', '.join(parts[1:])})")
                break

            if action == "join":
                new_coalition = frozenset(S | {player})
                if new_coalition in visited_coalitions:
                    if self.verbose:
                        print(f"  Step {step_num}: stopping (cycle detected on join)")
                    break

                S.add(player)
                v_S = utility_func(S)
                last_added = player
                greedy_order.append(player)
                scenario_values[player] += improvement
                visited_coalitions.add(new_coalition)

                process_history.append(
                    {
                        "step": step_num,
                        "action": "join",
                        "player": player,
                        "improvement": improvement,
                        "coalition": frozenset(S),
                        "coalition_size": len(S),
                        "value": v_S,
                        "transition_probs": step_transition_probs,
                    }
                )

                if self.verbose:
                    print(
                        f"  Step {step_num}: JOIN player "
                        f"{self.player_ids[player]} "
                        f"(improvement={improvement:.6f})"
                    )
            else:
                # LEAVE action
                new_coalition = frozenset(S - {player})
                if new_coalition in visited_coalitions:
                    if self.verbose:
                        print(f"  Step {step_num}: stopping (cycle detected on leave)")
                    break

                S.remove(player)
                v_S = utility_func(S)
                last_added = None
                if player in greedy_order:
                    greedy_order.remove(player)
                # Formula (4): contribution = v(S_{k+1}) - v(S_k)
                # For leave: v(S\{j}) - v(S) = improvement (positive when removal helped)
                scenario_values[player] += improvement
                visited_coalitions.add(new_coalition)

                process_history.append(
                    {
                        "step": step_num,
                        "action": "leave",
                        "player": player,
                        "improvement": improvement,
                        "coalition": frozenset(S),
                        "coalition_size": len(S),
                        "value": v_S,
                        "transition_probs": step_transition_probs,
                    }
                )

                if self.verbose:
                    print(
                        f"  Step {step_num}: LEAVE player "
                        f"{self.player_ids[player]} "
                        f"(improvement={improvement:.6f})"
                    )

        # Shapley values are the accumulated scenario-values (formula 4)
        shapley_values = scenario_values

        final_coalition = frozenset(S)
        return (
            shapley_values,
            greedy_order,
            final_coalition,
            marginal_records,
            process_history,
        )

    def _build_monotone_history(
        self,
        order: List[int],
        records: List[dict],
        utility_func,
        initial_set: Optional[Set[int]] = None,
    ) -> List[dict]:
        """Construct process_history from a completed monotone greedy process.

        Args:
            order: Greedy addition order (internal indices). If initial_set is
                provided, the first len(initial_set) entries are initial members.
            records: Marginal records from the greedy process.
            utility_func: Maps internal-index sets to coalition values.
            initial_set: Internal indices of pre-specified starting coalition.

        Returns:
            List of step dicts compatible with get_process_history().
        """
        history = []
        n_initial = len(initial_set) if initial_set else 0

        # Initial coalition members (first n_initial entries in order)
        if initial_set:
            S_init = set(initial_set)
            v_init = utility_func(S_init)
            for step_idx in range(n_initial):
                player = order[step_idx]
                mc = v_init - utility_func(S_init - {player})
                history.append(
                    {
                        "step": step_idx + 1,
                        "action": "initial",
                        "player": player,
                        "improvement": mc,
                        "coalition": frozenset(S_init),
                        "coalition_size": len(S_init),
                        "value": v_init,
                    }
                )

        # Greedily-added members
        S = set(initial_set) if initial_set else set()
        for step_idx, player in enumerate(order[n_initial:], start=n_initial):
            v_before = utility_func(S)
            S.add(player)
            v_after = utility_func(S)
            history.append(
                {
                    "step": step_idx + 1,
                    "action": "join",
                    "player": player,
                    "improvement": v_after - v_before,
                    "coalition": frozenset(S),
                    "coalition_size": len(S),
                    "value": v_after,
                }
            )
        return history

    def get_process_history(self) -> List[dict]:
        """Return the step-by-step process history of coalition formation.

        Each entry contains:
        - step: Step number (1-indexed)
        - action: 'join', 'leave', or 'initial'
        - player: Player identifier
        - improvement: Change in characteristic function value
        - coalition: Coalition after this step
        - coalition_size: Size of coalition after this step
        - value: Characteristic function value after this step

        Returns
        -------
        List[dict]
            Process history with player ids mapped back.

        Raises
        ------
        ValueError
            If explain() has not been called yet.
        """
        if self._process_history is None:
            raise ValueError("No process history available. Call explain() first.")

        # Map internal indices back to player ids
        mapped_history = []
        for entry in self._process_history:
            mapped_entry = dict(entry)
            mapped_entry["player"] = self.player_ids[entry["player"]]
            mapped_entry["coalition"] = frozenset(
                self.player_ids[i] for i in entry["coalition"]
            )
            mapped_history.append(mapped_entry)
        return mapped_history

    def build_interaction_graph(self) -> nx.Graph:
        """Build a feature interaction graph from greedy process marginals.

        For each pair (i, j), the interaction strength is defined as:
            |mean_marginal(i | j in S) - mean_marginal(i | j not in S)|
        symmetrized by averaging both directions.

        Returns
        -------
        nx.Graph
            Weighted interaction graph with player ids as nodes.

        Raises
        ------
        ValueError
            If explain() has not been called yet.
        """
        if self._marginal_records is None:
            raise ValueError("No marginal records available. Call explain() first.")

        # Group marginals by player and presence/absence of each other player
        # marginals_with[i][j] = list of marginals of i when j is in S
        # marginals_without[i][j] = list of marginals of i when j is not in S
        marginals_with = defaultdict(lambda: defaultdict(list))
        marginals_without = defaultdict(lambda: defaultdict(list))

        for rec in self._marginal_records:
            # Only use join-action records for interaction computation
            if rec.get("action", "join") != "join":
                continue
            i = rec["player"]
            coalition = rec["coalition"]
            mc = rec["marginal"]
            for j in range(self.n):
                if j == i:
                    continue
                if j in coalition:
                    marginals_with[i][j].append(mc)
                else:
                    marginals_without[i][j].append(mc)

        G = nx.Graph()
        G.add_nodes_from(self.player_ids)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Compute interaction(i, j)
                interaction_ij = self._pair_interaction(
                    marginals_with[i][j], marginals_without[i][j]
                )
                interaction_ji = self._pair_interaction(
                    marginals_with[j][i], marginals_without[j][i]
                )
                weight = (interaction_ij + interaction_ji) / 2.0

                if weight > 0:
                    G.add_edge(
                        self.player_ids[i],
                        self.player_ids[j],
                        weight=weight,
                    )

        return G

    @staticmethod
    def _pair_interaction(with_list: List[float], without_list: List[float]) -> float:
        """Compute |mean(with) - mean(without)| for a single direction."""
        mean_with = np.mean(with_list) if with_list else 0.0
        mean_without = np.mean(without_list) if without_list else 0.0
        return abs(mean_with - mean_without)

    def get_greedy_order(self) -> List:
        """Return the order in which players were added during the greedy process.

        Returns
        -------
        List
            Player ids in the order they were added.

        Raises
        ------
        ValueError
            If explain() has not been called yet.
        """
        if self._greedy_order is None:
            raise ValueError("No results available. Call explain() first.")
        return list(self._greedy_order)

    def get_final_coalition(self) -> frozenset:
        """Return the terminal coalition from the greedy process.

        Returns
        -------
        frozenset
            Set of player ids in the final coalition.

        Raises
        ------
        ValueError
            If explain() has not been called yet.
        """
        if self._final_coalition is None:
            raise ValueError("No results available. Call explain() first.")
        return self._final_coalition
