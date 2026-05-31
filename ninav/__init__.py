"""ninav -- a clean, tested reference implementation of neural inertial navigation.

A from-scratch rebuild of the ``inertial_navigation_transformer`` course project,
fixing the fundamental "compression" bugs documented in ``REVIEW.md`` and following
the RoNIN / TLIO / IONet literature: regress a heading-agnostic, gravity-aligned
2D velocity / displacement from short IMU windows, then integrate.
"""

__version__ = "0.1.0"
