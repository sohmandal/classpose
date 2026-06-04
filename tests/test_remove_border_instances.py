import numpy as np
import pytest

from classpose.metrics.pq import remove_border_instances


def _make_instance_mask():
    """
    Create a 6x6 instance-only mask (H, W) with 4 instances.

    Layout (instance IDs):
        - Instance 1: top-left 3x3 block  → touches top and left borders
        - Instance 2: top-right 3x3 block → touches top and right borders
        - Instance 3: centre 2x2 block    → does NOT touch any border
        - Instance 4: bottom-right 3x3 block → touches bottom and right borders
    """
    mask = np.zeros((6, 6), dtype=np.int64)
    mask[0:3, 0:3] = 1
    mask[0:3, 3:6] = 2
    mask[2:4, 2:4] = 3
    mask[3:6, 3:6] = 4
    return mask


class TestRemoveBorderInstancesInstanceOnly:
    """
    Tests for 2-D instance-only masks (H, W).
    """

    def test_border_instances_are_zeroed(self):
        mask = _make_instance_mask()
        result = remove_border_instances(mask)
        # Instances 1, 2, 4 touch the border and should be removed
        assert np.all(result[result != 0] == 3)

    def test_interior_instance_is_preserved(self):
        mask = _make_instance_mask()
        result = remove_border_instances(mask)
        expected_results = [
            ([2, 2], 3),
            ([2, 3], 3),
            ([3, 2], 3),
            ([3, 3], 0),
        ]
        for (x, y), r in expected_results:
            assert result[x, y] == r

    def test_all_border_returns_all_zeros(self):
        """A mask where every instance touches the border."""
        mask = np.zeros((4, 4), dtype=np.int64)
        mask[0:2, :] = 1
        mask[2:4, :] = 2
        result = remove_border_instances(mask)
        assert np.all(result == 0)

    def test_empty_mask_is_unchanged(self):
        mask = np.zeros((5, 5), dtype=np.int64)
        result = remove_border_instances(mask)
        assert np.all(result == 0)

    def test_single_interior_instance(self):
        mask = np.zeros((5, 5), dtype=np.int64)
        mask[1:4, 1:4] = 7
        result = remove_border_instances(mask)
        assert np.all(result[1:4, 1:4] == 7)
        assert result[0, :].sum() == 0
        assert result[:, 0].sum() == 0


class TestRemoveBorderInstancesWithClass:
    """
    Tests for 3-D masks (H, W, 2) with instance + class channels.
    """

    def _make_combined_mask(self):
        """Build an (H, W, 2) mask from the instance layout."""
        inst = _make_instance_mask()
        cls = np.zeros_like(inst)
        cls[inst == 1] = 1
        cls[inst == 2] = 2
        cls[inst == 3] = 3
        cls[inst == 4] = 1
        return np.stack([inst, cls], axis=-1)

    def test_border_instances_zeroed_both_channels(self):
        mask = self._make_combined_mask()
        result = remove_border_instances(mask)
        # Only instance 3 should survive in both channels
        surviving_inst = np.unique(result[..., 0])
        assert set(surviving_inst) == {0, 3}
        surviving_cls = np.unique(result[..., 1])
        assert set(surviving_cls) == {0, 3}

    def test_interior_instance_class_preserved(self):
        mask = self._make_combined_mask()
        result = remove_border_instances(mask)
        expected_results = [
            ([2, 2, 0], 3),
            ([2, 3, 0], 3),
            ([3, 2, 0], 3),
            ([3, 3, 0], 0),
            ([2, 2, 1], 3),
            ([2, 3, 1], 3),
            ([3, 2, 1], 3),
            ([3, 3, 1], 0),
        ]
        for (x, y, c), r in expected_results:
            assert result[x, y, c] == r

    def test_all_border_returns_all_zeros(self):
        inst = np.zeros((4, 4), dtype=np.int64)
        inst[0:2, :] = 1
        inst[2:4, :] = 2
        cls = np.ones_like(inst)
        mask = np.stack([inst, cls], axis=-1)
        result = remove_border_instances(mask)
        assert np.all(result == 0)
