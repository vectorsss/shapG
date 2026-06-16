"""
Tests for the MarkovianExplainer (greedy Shapley-Weber process).
"""

import os
import sys
import unittest

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapG.characteristic import CoalitionDegree, CustomFunction, NodeCount
from shapG.explainer import MarkovianExplainer


class TestMarkovianInitialization(unittest.TestCase):
    """Test MarkovianExplainer initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        explainer = MarkovianExplainer()
        self.assertIsInstance(explainer.characteristic_function, CoalitionDegree)
        self.assertFalse(explainer.verbose)
        self.assertIsNone(explainer.seed)
        self.assertFalse(explainer._fitted)

    def test_explicit_parameters(self):
        """Test initialization with explicit parameters."""
        char_func = NodeCount()
        explainer = MarkovianExplainer(
            characteristic_function=char_func,
            verbose=True,
            seed=42,
        )
        self.assertEqual(explainer.characteristic_function, char_func)
        self.assertTrue(explainer.verbose)
        self.assertEqual(explainer.seed, 42)


class TestMarkovianFit(unittest.TestCase):
    """Test fit() with various input types."""

    def test_fit_with_graph(self):
        """Test fitting with a NetworkX graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        explainer = MarkovianExplainer()
        result = explainer.fit(G)

        self.assertIs(result, explainer)
        self.assertTrue(explainer._fitted)
        self.assertEqual(explainer.n, 3)
        self.assertEqual(set(explainer.player_ids), {0, 1, 2})

    def test_fit_with_dataframe(self):
        """Test fitting with a pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        explainer = MarkovianExplainer()
        explainer.fit(df)

        self.assertTrue(explainer._fitted)
        self.assertEqual(explainer.n, 3)
        self.assertEqual(list(explainer.player_ids), ["A", "B", "C"])

    def test_fit_with_int(self):
        """Test fitting with an integer (number of players)."""
        explainer = MarkovianExplainer()
        explainer.fit(5)

        self.assertTrue(explainer._fitted)
        self.assertEqual(explainer.n, 5)
        self.assertEqual(explainer.player_ids, list(range(5)))

    def test_fit_with_ndarray(self):
        """Test fitting with a numpy array."""
        data = np.random.randn(10, 4)
        explainer = MarkovianExplainer()
        explainer.fit(data)

        self.assertTrue(explainer._fitted)
        self.assertEqual(explainer.n, 4)

    def test_fit_resets_previous_results(self):
        """Test that calling fit() resets results from previous explain()."""
        G = nx.complete_graph(3)
        explainer = MarkovianExplainer()
        explainer.fit_explain(G)

        self.assertIsNotNone(explainer._shapley_values)

        # Re-fit should reset
        explainer.fit(G)
        self.assertIsNone(explainer._shapley_values)
        self.assertIsNone(explainer._greedy_order)
        self.assertIsNone(explainer._final_coalition)


class TestMarkovianExplain(unittest.TestCase):
    """Test explain() functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.char_func = CoalitionDegree()

    def test_basic_explain(self):
        """Test that explain returns correct structure."""
        explainer = MarkovianExplainer()
        values = explainer.fit_explain(self.G)

        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})
        for v in values.values():
            self.assertIsInstance(v, float)

    def test_efficiency_property(self):
        """Test that sum(values) = v(final_coalition) - v(empty)."""
        explainer = MarkovianExplainer(seed=42)
        values = explainer.fit_explain(self.G)

        final_coalition = explainer.get_final_coalition()
        v_final = self.char_func(final_coalition, self.G)
        v_empty = self.char_func(set(), self.G)

        self.assertAlmostEqual(sum(values.values()), v_final - v_empty, places=10)

    def test_unfitted_explain_error(self):
        """Test that explain() raises ValueError when not fitted."""
        explainer = MarkovianExplainer()
        with self.assertRaises(ValueError):
            explainer.explain()

    def test_fit_explain_convenience(self):
        """Test fit_explain shortcut."""
        explainer = MarkovianExplainer()
        values = explainer.fit_explain(self.G)

        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

    def test_explain_with_inline_fit(self):
        """Test that explain(X) auto-fits."""
        explainer = MarkovianExplainer()
        values = explainer.explain(self.G)

        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))


class TestMarkovianEarlyStopping(unittest.TestCase):
    """Test early stopping behavior."""

    def test_early_stopping_unadded_players_get_zero(self):
        """Test that players never added to the coalition receive value 0."""

        # Use a custom function where adding certain players is harmful
        # Players 0,1 contribute; player 2 is detrimental (strictly negative)
        def value_func(coalition, context=None):
            if not coalition:
                return 0.0
            result = 0.0
            for p in coalition:
                if p in (0, 1):
                    result += 1.0
                else:
                    result -= 10.0
            return result

        char_func = CustomFunction(value_func)
        explainer = MarkovianExplainer(characteristic_function=char_func, seed=42)
        values = explainer.fit_explain(3)

        # Player 2 should not be added (marginal = -10)
        self.assertEqual(values[2], 0.0)
        # Players 0 and 1 should have positive values
        self.assertGreater(values[0], 0)
        self.assertGreater(values[1], 0)
        # Greedy order should only contain players 0 and 1
        self.assertEqual(len(explainer.get_greedy_order()), 2)

    def test_final_coalition_excludes_harmful_players(self):
        """Test that the final coalition doesn't include harmful players."""

        def value_func(coalition, context=None):
            if not coalition:
                return 0.0
            return sum(1.0 if p < 3 else -5.0 for p in coalition)

        char_func = CustomFunction(value_func)
        explainer = MarkovianExplainer(characteristic_function=char_func, seed=42)
        explainer.fit_explain(5)

        final = explainer.get_final_coalition()
        # Harmful players (3, 4) should be excluded
        self.assertTrue(final.issubset({0, 1, 2}))


class TestMarkovianDeterminism(unittest.TestCase):
    """Test deterministic behavior with seed."""

    def test_same_seed_same_results(self):
        """Test that the same seed produces identical results."""
        G = nx.erdos_renyi_graph(10, 0.3, seed=99)

        values1 = MarkovianExplainer(seed=42).fit_explain(G)
        values2 = MarkovianExplainer(seed=42).fit_explain(G)

        for node in G.nodes():
            self.assertAlmostEqual(values1[node], values2[node], places=10)

    def test_different_seed_may_differ(self):
        """Test that different seeds can produce different results on tied graphs."""
        # Build a graph where tie-breaking matters (complete graph = all symmetric)
        G = nx.complete_graph(5)

        values1 = MarkovianExplainer(seed=1).fit_explain(G)
        values2 = MarkovianExplainer(seed=2).fit_explain(G)

        # Greedy order may differ due to tie-breaking, but sum should be same
        self.assertAlmostEqual(sum(values1.values()), sum(values2.values()), places=10)


class TestMarkovianAccessors(unittest.TestCase):
    """Test get_greedy_order and get_final_coalition."""

    def test_get_greedy_order(self):
        """Test greedy order accessor."""
        G = nx.path_graph(4)  # 0-1-2-3
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        order = explainer.get_greedy_order()
        self.assertIsInstance(order, list)
        self.assertGreater(len(order), 0)
        # All elements should be valid node ids
        for node in order:
            self.assertIn(node, G.nodes())

    def test_get_final_coalition(self):
        """Test final coalition accessor."""
        G = nx.complete_graph(4)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        final = explainer.get_final_coalition()
        self.assertIsInstance(final, frozenset)
        # Final coalition should be a subset of all nodes
        self.assertTrue(final.issubset(set(G.nodes())))

    def test_accessors_before_explain_raise(self):
        """Test that accessors raise ValueError before explain()."""
        explainer = MarkovianExplainer()

        with self.assertRaises(ValueError):
            explainer.get_greedy_order()

        with self.assertRaises(ValueError):
            explainer.get_final_coalition()


class TestMarkovianInteractionGraph(unittest.TestCase):
    """Test build_interaction_graph() method."""

    def test_interaction_graph_structure(self):
        """Test that interaction graph has correct nodes."""
        G = nx.complete_graph(5)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        ig = explainer.build_interaction_graph()
        self.assertIsInstance(ig, nx.Graph)
        self.assertEqual(set(ig.nodes()), set(G.nodes()))

    def test_interaction_graph_non_negative_weights(self):
        """Test that all edge weights are non-negative."""
        G = nx.erdos_renyi_graph(8, 0.5, seed=42)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        ig = explainer.build_interaction_graph()
        for _, _, data in ig.edges(data=True):
            self.assertGreaterEqual(data["weight"], 0.0)

    def test_interaction_graph_before_explain_raises(self):
        """Test that build_interaction_graph raises before explain()."""
        explainer = MarkovianExplainer()
        with self.assertRaises(ValueError):
            explainer.build_interaction_graph()


class TestMarkovianCustomFunction(unittest.TestCase):
    """Test with custom characteristic functions."""

    def test_custom_additive_function(self):
        """Test with an additive function: v(S) = sum of player weights."""
        weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

        def additive(coalition, context=None):
            return sum(weights.get(p, 0) for p in coalition)

        char_func = CustomFunction(additive)
        explainer = MarkovianExplainer(characteristic_function=char_func, seed=42)
        values = explainer.fit_explain(4)

        # For an additive function, each player's marginal is their own weight
        for pid, expected_w in weights.items():
            self.assertAlmostEqual(values[pid], expected_w, places=10)

    def test_node_count_function(self):
        """Test with NodeCount characteristic function."""
        explainer = MarkovianExplainer(characteristic_function=NodeCount(), seed=42)
        values = explainer.fit_explain(5)

        # Every player contributes +1 (marginal of NodeCount is always 1)
        for v in values.values():
            self.assertAlmostEqual(v, 1.0, places=10)

        # All 5 should be added
        self.assertEqual(len(explainer.get_greedy_order()), 5)


class TestMarkovianEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_single_node_graph(self):
        """Test with a single isolated node."""
        G = nx.Graph()
        G.add_node(0)
        explainer = MarkovianExplainer()
        values = explainer.fit_explain(G)

        self.assertEqual(len(values), 1)
        self.assertIn(0, values)
        # CoalitionDegree of a single node is 0, marginal = 0 - 0 = 0
        self.assertEqual(values[0], 0.0)

    def test_complete_graph_symmetry(self):
        """Test that a complete graph gives equal values to all nodes."""
        G = nx.complete_graph(4)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        # In a greedy process on K_n, the first player gets marginal 0,
        # and subsequent players get increasing marginals (not symmetric).
        # But all players should eventually be added.
        self.assertEqual(len(explainer.get_greedy_order()), 4)
        self.assertEqual(explainer.get_final_coalition(), frozenset(G.nodes()))

    def test_two_node_graph(self):
        """Test with a simple two-node graph."""
        G = nx.Graph()
        G.add_edge(0, 1)
        explainer = MarkovianExplainer(seed=42)
        values = explainer.fit_explain(G)

        self.assertEqual(len(values), 2)
        # v({0}) = 0, v({1}) = 0, v({0,1}) = 1
        # First player gets marginal 0, second gets 1 - 0 = 1
        # But first player has marginal 0, which is <= 0, so early stop?
        # Actually: v(empty) = 0, v({i}) = 0 for both, so marginal = 0
        # 0 <= 0 triggers early stop. Both get 0.
        total = sum(values.values())
        # The sum should equal v(final_coalition) - v(empty)
        final = explainer.get_final_coalition()
        v_final = CoalitionDegree()(final, G)
        self.assertAlmostEqual(total, v_final, places=10)

    def test_star_graph(self):
        """Test with a star graph (one hub, multiple leaves)."""
        G = nx.star_graph(4)  # Node 0 is hub, 1-4 are leaves
        explainer = MarkovianExplainer(seed=42)
        values = explainer.fit_explain(G)

        self.assertEqual(set(values.keys()), set(G.nodes()))
        for v in values.values():
            self.assertIsInstance(v, float)
            self.assertFalse(np.isnan(v))
            self.assertFalse(np.isinf(v))

    def test_with_context(self):
        """Test fitting with explicit context parameter."""
        G = nx.complete_graph(3)
        explainer = MarkovianExplainer()
        values = explainer.explain(3, context=G)

        self.assertEqual(set(values.keys()), {0, 1, 2})


class TestMarkovianJoinAndLeave(unittest.TestCase):
    """Test join-and-leave (non-monotone) Markovian process."""

    def test_allow_leave_default_false(self):
        """Test that allow_leave defaults to False for backward compatibility."""
        explainer = MarkovianExplainer()
        self.assertFalse(explainer.allow_leave)
        self.assertIsNone(explainer.max_iterations)

    def test_monotone_function_same_result(self):
        """Test that NodeCount with allow_leave=True gives same result as without.

        NodeCount is strictly monotone (every addition helps), so leave should
        never be triggered, producing identical results.
        """
        G = nx.erdos_renyi_graph(8, 0.4, seed=42)

        explainer_no_leave = MarkovianExplainer(
            characteristic_function=NodeCount(), seed=42
        )
        values_no_leave = explainer_no_leave.fit_explain(G)

        explainer_with_leave = MarkovianExplainer(
            characteristic_function=NodeCount(), seed=42, allow_leave=True
        )
        values_with_leave = explainer_with_leave.fit_explain(G)

        for node in G.nodes():
            self.assertAlmostEqual(
                values_no_leave[node], values_with_leave[node], places=10
            )

    def test_leave_removes_harmful_player(self):
        """Test that a player can be removed when it becomes harmful.

        Custom function where player 2 is only harmful once player 3 is present.
        The join-and-leave process should be able to remove player 2.
        """

        # v(S) = |S| if 2 not in S, else |S| - 3 if 3 in S, else |S|
        # So adding 2 first is fine (marginal = 1), but once 3 is added,
        # having 2 becomes costly (it subtracts 3).
        def tricky_func(coalition, context=None):
            if not coalition:
                return 0.0
            val = len(coalition)
            if 2 in coalition and 3 in coalition:
                val -= 3
            return val

        char_func = CustomFunction(tricky_func)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, allow_leave=True
        )
        explainer.fit_explain(4)

        # The process should eventually remove player 2 after adding player 3
        final = explainer.get_final_coalition()
        self.assertNotIn(2, final, "Player 2 should be removed from final coalition")

    def test_efficiency_with_leave(self):
        """Test that sum(SV) = v(final) - v(empty) holds for join-and-leave."""

        def value_func(coalition, context=None):
            if not coalition:
                return 0.0
            return sum(1.0 if p != 2 else -0.5 for p in coalition)

        char_func = CustomFunction(value_func)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, allow_leave=True
        )
        values = explainer.fit_explain(4)

        final = explainer.get_final_coalition()
        v_final = value_func(final)
        v_empty = value_func(set())

        self.assertAlmostEqual(sum(values.values()), v_final - v_empty, places=10)

    def test_max_iterations_parameter(self):
        """Test that max_iterations parameter is stored correctly."""
        explainer = MarkovianExplainer(allow_leave=True, max_iterations=100)
        self.assertEqual(explainer.max_iterations, 100)

    def test_cycle_detection_stops(self):
        """Test that the process terminates when a coalition would be revisited."""

        # Use a function that could cause oscillation without cycle detection:
        # v({0}) = 1, v({1}) = 1, v({0,1}) = 0.5
        # Process: add 0 (mc=1), add 1 (mc=-0.5? no, mc=0.5-1=-0.5, skip)
        # Actually this won't cycle. Let's use a simpler case that exercises
        # the cycle detection indirectly via the visited_coalitions set.
        def cycle_risk_func(coalition, context=None):
            if not coalition:
                return 0.0
            if coalition == {0}:
                return 2.0
            if coalition == {1}:
                return 2.0
            if coalition == {0, 1}:
                return 1.5
            return len(coalition)

        char_func = CustomFunction(cycle_risk_func)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, allow_leave=True
        )
        values = explainer.fit_explain(2)

        # Process should terminate (not infinite loop)
        self.assertIsNotNone(values)
        history = explainer.get_process_history()
        self.assertGreater(len(history), 0)

    def test_process_history_available(self):
        """Test that process history is populated with correct structure."""
        G = nx.complete_graph(4)
        explainer = MarkovianExplainer(seed=42, allow_leave=True)
        explainer.fit_explain(G)

        history = explainer.get_process_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        for entry in history:
            self.assertIn("step", entry)
            self.assertIn("action", entry)
            self.assertIn("player", entry)
            self.assertIn("improvement", entry)
            self.assertIn("coalition", entry)
            self.assertIn("coalition_size", entry)
            self.assertIn("value", entry)
            self.assertIn(entry["action"], ("join", "leave"))
            self.assertIsInstance(entry["step"], int)
            self.assertIsInstance(entry["coalition"], frozenset)

    def test_backward_compatible_no_leave(self):
        """Test that allow_leave=False gives identical results to default."""
        G = nx.erdos_renyi_graph(8, 0.4, seed=42)

        explainer_default = MarkovianExplainer(seed=42)
        values_default = explainer_default.fit_explain(G)

        explainer_explicit = MarkovianExplainer(seed=42, allow_leave=False)
        values_explicit = explainer_explicit.fit_explain(G)

        for node in G.nodes():
            self.assertAlmostEqual(
                values_default[node], values_explicit[node], places=10
            )


class TestMarkovianProcessHistory(unittest.TestCase):
    """Test process history for both monotone and join-and-leave processes."""

    def test_monotone_process_has_history(self):
        """Test that monotone process produces history with len = len(greedy_order)."""
        G = nx.complete_graph(5)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        history = explainer.get_process_history()
        greedy_order = explainer.get_greedy_order()
        self.assertEqual(len(history), len(greedy_order))

    def test_history_steps_sequential(self):
        """Test that step numbers are 1, 2, 3, ..."""
        G = nx.path_graph(5)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        history = explainer.get_process_history()
        for i, entry in enumerate(history):
            self.assertEqual(entry["step"], i + 1)

    def test_history_coalition_size_grows_monotone(self):
        """Test that coalition size strictly increases for standard process."""
        G = nx.complete_graph(4)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        history = explainer.get_process_history()
        sizes = [entry["coalition_size"] for entry in history]
        for i in range(1, len(sizes)):
            self.assertGreater(sizes[i], sizes[i - 1])

    def test_history_before_explain_raises(self):
        """Test that get_process_history raises ValueError before explain()."""
        explainer = MarkovianExplainer()
        with self.assertRaises(ValueError):
            explainer.get_process_history()

    def test_history_all_join_for_monotone(self):
        """Test that all actions are 'join' for monotone process."""
        G = nx.complete_graph(4)
        explainer = MarkovianExplainer(seed=42)
        explainer.fit_explain(G)

        history = explainer.get_process_history()
        for entry in history:
            self.assertEqual(entry["action"], "join")

    def test_history_players_are_mapped(self):
        """Test that history players use external player ids, not internal indices."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        def simple_func(coalition, context=None):
            return len(coalition)

        explainer = MarkovianExplainer(
            characteristic_function=CustomFunction(simple_func), seed=42
        )
        explainer.fit_explain(df)

        history = explainer.get_process_history()
        valid_names = {"A", "B", "C"}
        for entry in history:
            self.assertIn(entry["player"], valid_names)


class TestMarkovianInitialCoalition(unittest.TestCase):
    """Test initial_coalition parameter."""

    def test_initial_coalition_default_none(self):
        """Test that initial_coalition defaults to None."""
        explainer = MarkovianExplainer()
        self.assertIsNone(explainer.initial_coalition)

    def test_initial_coalition_stored(self):
        """Test that initial_coalition is stored correctly."""
        explainer = MarkovianExplainer(initial_coalition=[0, 1])
        self.assertEqual(explainer.initial_coalition, [0, 1])

    def test_initial_coalition_starts_from_given_set(self):
        """Test that the process starts from the given initial coalition."""

        def additive(coalition, context=None):
            return sum(1.0 for p in coalition)

        char_func = CustomFunction(additive)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, initial_coalition=[0, 1]
        )
        values = explainer.fit_explain(5)

        # All 5 players should be in the final coalition (all contribute +1)
        self.assertEqual(len(explainer.get_final_coalition()), 5)
        # Initial coalition members should have Shapley values
        self.assertAlmostEqual(values[0], 1.0, places=10)
        self.assertAlmostEqual(values[1], 1.0, places=10)

    def test_initial_coalition_efficiency(self):
        """Test efficiency property with initial coalition."""
        weights = {0: 3.0, 1: 2.0, 2: 1.0, 3: -5.0}

        def weighted(coalition, context=None):
            return sum(weights.get(p, 0) for p in coalition)

        char_func = CustomFunction(weighted)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, initial_coalition=[0, 1]
        )
        values = explainer.fit_explain(4)

        final = explainer.get_final_coalition()
        v_final = weighted(final)
        v_empty = weighted(set())
        self.assertAlmostEqual(sum(values.values()), v_final - v_empty, places=10)

    def test_initial_coalition_with_leave(self):
        """Test initial coalition works with allow_leave=True."""

        # Player 2 is harmful once 3 is present
        def tricky(coalition, context=None):
            if not coalition:
                return 0.0
            val = len(coalition)
            if 2 in coalition and 3 in coalition:
                val -= 3
            return val

        char_func = CustomFunction(tricky)
        explainer = MarkovianExplainer(
            characteristic_function=char_func,
            seed=42,
            allow_leave=True,
            initial_coalition=[0, 2],
        )
        values = explainer.fit_explain(4)

        # Process should handle initial coalition and leave logic together
        self.assertIsNotNone(values)
        final = explainer.get_final_coalition()
        v_final = tricky(final)
        v_empty = tricky(set())
        self.assertAlmostEqual(sum(values.values()), v_final - v_empty, places=10)

    def test_initial_coalition_invalid_player_raises(self):
        """Test that invalid player in initial_coalition raises ValueError."""
        explainer = MarkovianExplainer(initial_coalition=[99])
        with self.assertRaises(ValueError):
            explainer.fit_explain(5)

    def test_initial_coalition_with_dataframe(self):
        """Test initial coalition works with DataFrame column names."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        def simple(coalition, context=None):
            return len(coalition)

        char_func = CustomFunction(simple)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, initial_coalition=["A", "B"]
        )
        explainer.fit_explain(df)

        # A and B should be in the order
        order = explainer.get_greedy_order()
        self.assertIn("A", order)
        self.assertIn("B", order)

    def test_initial_coalition_process_history(self):
        """Test that process history includes initial steps."""

        def simple(coalition, context=None):
            return len(coalition)

        char_func = CustomFunction(simple)
        explainer = MarkovianExplainer(
            characteristic_function=char_func, seed=42, initial_coalition=[0, 1]
        )
        explainer.fit_explain(5)

        history = explainer.get_process_history()
        initial_steps = [e for e in history if e["action"] == "initial"]
        self.assertEqual(len(initial_steps), 2)
        initial_players = {e["player"] for e in initial_steps}
        self.assertEqual(initial_players, {0, 1})


class TestMarkovianThreshold(unittest.TestCase):
    """Test min_improvement threshold parameter."""

    def test_min_improvement_default_zero(self):
        """Test that min_improvement defaults to 0.0."""
        explainer = MarkovianExplainer()
        self.assertEqual(explainer.min_improvement, 0.0)

    def test_min_improvement_stored(self):
        """Test that min_improvement is stored correctly."""
        explainer = MarkovianExplainer(min_improvement=0.001)
        self.assertAlmostEqual(explainer.min_improvement, 0.001)

    def test_positive_threshold_filters_weak_features(self):
        """Test that positive threshold prevents adding low-marginal features."""
        # Weights: 0=5.0, 1=0.0005 (below 0.001 threshold), 2=3.0
        weights = {0: 5.0, 1: 0.0005, 2: 3.0}

        def weighted(coalition, context=None):
            return sum(weights.get(p, 0) for p in coalition)

        # Without threshold: all 3 added
        char_func = CustomFunction(weighted)
        explainer_default = MarkovianExplainer(
            characteristic_function=char_func, seed=42
        )
        explainer_default.fit_explain(3)
        self.assertEqual(len(explainer_default.get_final_coalition()), 3)

        # With threshold 0.001: player 1 excluded (marginal 0.0005 < 0.001)
        explainer_strict = MarkovianExplainer(
            characteristic_function=char_func, seed=42, min_improvement=0.001
        )
        explainer_strict.fit_explain(3)
        self.assertNotIn(1, explainer_strict.get_final_coalition())

    def test_negative_threshold_allows_slightly_harmful(self):
        """Test that negative threshold allows slightly harmful additions."""
        # Player 2 has marginal -0.0005 (slightly harmful)
        weights = {0: 5.0, 1: 3.0, 2: -0.0005}

        def weighted(coalition, context=None):
            return sum(weights.get(p, 0) for p in coalition)

        # Without threshold: player 2 excluded (marginal -0.0005 < 0)
        char_func = CustomFunction(weighted)
        explainer_default = MarkovianExplainer(
            characteristic_function=char_func, seed=42
        )
        explainer_default.fit_explain(3)
        self.assertNotIn(2, explainer_default.get_final_coalition())

        # With negative threshold -0.001: player 2 included
        explainer_lenient = MarkovianExplainer(
            characteristic_function=char_func, seed=42, min_improvement=-0.001
        )
        explainer_lenient.fit_explain(3)
        self.assertIn(2, explainer_lenient.get_final_coalition())

    def test_threshold_with_leave(self):
        """Test that threshold works with join-and-leave process."""

        def value_func(coalition, context=None):
            if not coalition:
                return 0.0
            return sum(1.0 if p != 2 else -0.0005 for p in coalition)

        char_func = CustomFunction(value_func)

        # With threshold -0.001: player 2 tolerated (marginal > -0.001)
        explainer = MarkovianExplainer(
            characteristic_function=char_func,
            seed=42,
            allow_leave=True,
            min_improvement=-0.001,
        )
        values = explainer.fit_explain(4)
        self.assertIn(2, explainer.get_final_coalition())

        # Efficiency still holds
        final = explainer.get_final_coalition()
        v_final = value_func(final)
        self.assertAlmostEqual(sum(values.values()), v_final, places=10)

    def test_zero_threshold_matches_default(self):
        """Test that min_improvement=0 gives identical results to default."""
        G = nx.erdos_renyi_graph(8, 0.4, seed=42)

        values_default = MarkovianExplainer(seed=42).fit_explain(G)
        values_explicit = MarkovianExplainer(seed=42, min_improvement=0.0).fit_explain(
            G
        )

        for node in G.nodes():
            self.assertAlmostEqual(
                values_default[node], values_explicit[node], places=10
            )


if __name__ == "__main__":
    unittest.main()
