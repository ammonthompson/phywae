import re
import sys
import unittest
from pathlib import Path

import numpy as np
import dendropy as dp
from typing import Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Test runner may not install the package; allow importing from src/.
sys.path.insert(0, str(_repo_root() / "src"))

from phywae import utils  # noqa: E402


def _distance_matrix(tree: dp.Tree, tip_labels: List[str]) -> np.ndarray:
    pdm = tree.phylogenetic_distance_matrix()
    taxon_by_label = {taxon.label: taxon for taxon in tree.taxon_namespace}
    matrix = np.zeros((len(tip_labels), len(tip_labels)), dtype=float)
    for i, a in enumerate(tip_labels):
        for j, b in enumerate(tip_labels):
            matrix[i, j] = float(pdm.distance(taxon_by_label[a], taxon_by_label[b]))
    return matrix


_TIP_ANN_RE = re.compile(r"(t\d+)(?::[^\[\),]+)?\[&([^\]]+)\]")


def _parse_tip_annotations(newick: str) -> Dict[str, Dict[str, float]]:
    annotations: Dict[str, Dict[str, float]] = {}
    for match in _TIP_ANN_RE.finditer(newick):
        tip_label = match.group(1)
        payload = match.group(2)
        values: Dict[str, float] = {}
        for kv in payload.split(","):
            key, value = kv.split("=", 1)
            values[key] = float(value.strip('"'))
        annotations[tip_label] = values
    return annotations


class TestConvertToNewick(unittest.TestCase):
    def test_simple_4tip_tree_roundtrip_and_annotations(self):
        # Use distinct root distances to avoid Phyddle reordering ties.
        nwk = "((t0:0.5,t1:0.1):0.2,(t2:0.4,t3:0.1):0.1);"
        tree = dp.Tree.get(data=nwk, schema="newick", rooting="force-rooted")

        # A small "dat matrix": rows are characters, columns are taxa.
        dat = {
            "t0": [0.111, 1.234],
            "t1": [0.222, 2.345],
            "t2": [0.333, 3.456],
            "t3": [0.444, 4.567],
        }

        # Captured from Phyddle 0.3.0 encode_cblvs(tree, dat, tree_width=4,
        # tree_encode_type="height_only", rescale=False) for the Newick above.
        encoded = np.array(
            [
                [0.7, 0.0, 0.111, 1.234],
                [0.10000000000000003, 0.2, 0.222, 2.345],
                [0.5, 0.0, 0.333, 3.456],
                [0.1, 0.1, 0.444, 4.567],
            ],
            dtype=float,
        )
        cblv = encoded[:, 0:2].T.reshape(1, 2, 4)
        char_data = encoded[:, 2:].T.reshape(1, 2, 4)
        num_tips = np.array([4.0], dtype=float)

        newicks = utils.convert_to_newick(cblv, num_tips, char_data)
        self.assertEqual(len(newicks), 1)
        decoded_tree = dp.Tree.get(data=newicks[0], schema="newick", rooting="force-rooted")

        tip_labels = ["t0", "t1", "t2", "t3"]
        np.testing.assert_allclose(
            _distance_matrix(decoded_tree, tip_labels),
            _distance_matrix(tree, tip_labels),
            rtol=1e-10,
            atol=1e-10,
        )

        observed = _parse_tip_annotations(newicks[0])
        self.assertEqual(set(observed.keys()), set(tip_labels))
        for tip in tip_labels:
            expected_values = np.round(dat[tip], 2)
            self.assertAlmostEqual(observed[tip]["char_0"], float(expected_values[0]), places=6)
            self.assertAlmostEqual(observed[tip]["char_1"], float(expected_values[1]), places=6)


if __name__ == "__main__":
    unittest.main()
