"""Environment configuration required before importing the solver.

Import this module *before* ``payne_zero_atmosphere`` (or anything that pulls in
Numba). It exists because the default Numba threading layer segfaults on macOS
when a second OpenMP runtime is already loaded, which is the normal situation
when the interpreter also has PyTorch or an Anaconda-linked NumPy in it. The
crash lands inside the ``parallel=True`` opacity kernels during the first
iteration, with no Python-level traceback.

``workqueue`` is Numba's OpenMP-free threading layer. It is slightly slower than
``omp`` but it is the only layer that survives a duplicated OpenMP runtime, and
it does not change results: chunked reductions are grouped by thread count, and
1-thread versus 8-thread solves agree bit-for-bit except for ``column_mass``,
which moves at the 1e-8 level from float reassociation.
"""

from __future__ import annotations

import os


DEFAULT_THREADING_LAYER = "workqueue"


def configure(threading_layer: str | None = None) -> dict[str, str]:
    """Set the Numba environment for a solver run and return what was applied.

    Existing values win, so an explicit ``NUMBA_THREADING_LAYER=omp`` in the
    caller's environment is honoured rather than silently overridden.
    """

    applied = {}
    layer = threading_layer or DEFAULT_THREADING_LAYER
    if "NUMBA_THREADING_LAYER" not in os.environ:
        os.environ["NUMBA_THREADING_LAYER"] = layer
    applied["NUMBA_THREADING_LAYER"] = os.environ["NUMBA_THREADING_LAYER"]
    if "NUMBA_NUM_THREADS" in os.environ:
        applied["NUMBA_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"]
    return applied


# Applied on import: callers only have to import this module early enough.
configure()
