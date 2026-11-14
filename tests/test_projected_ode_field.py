"""Check that the projected-field figure uses the intended curl sign."""

import numpy as np

from scripts.plot_odebrain_six_class_field import planar_curl


def test_projected_planar_curl_distinguishes_rotation_from_gradient():
    x = np.linspace(-1., 1., 11)
    y = np.linspace(-1., 1., 11)
    xx, yy = np.meshgrid(x, y)
    rotating = planar_curl(-yy, xx, x, y)
    gradient = planar_curl(xx, yy, x, y)
    assert np.allclose(rotating, 2.)
    assert np.allclose(gradient, 0.)
