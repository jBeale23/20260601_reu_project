"""Candidate models under evaluation. Nothing here is on the production path.

Every module in this package is a *challenger*: it is measured head-to-head against the
classifier in :mod:`domain_layout.profiles` on identical proteins and identical labels, and
it replaces nothing unless it wins on that measurement. Importing this package has no
effect on any existing result, and :mod:`domain_layout` never imports from it - a rule
enforced by ``tests/test_experimental_isolation.py`` rather than left to discipline.
"""
