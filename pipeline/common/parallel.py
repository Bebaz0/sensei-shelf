"""Order-preserving parallel map, so every caller gives exactly the output of its serial loop.

tmap: threads, for OpenCV image work (cv2 releases the GIL). pmap: forked processes, for pure-Python or sklearn work;
fn and items are inherited by the fork, only indices and results are pickled, so fn may be a closure.
WORKERS=1 in the environment makes both serial. pmap is serial off Linux (fork is unsafe with macOS Accelerate)
and inside a pmap worker (no nested pools).
"""
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

WORKERS = max(1, int(os.environ.get("WORKERS", os.cpu_count() or 1)))

_job: tuple | None = None
_in_worker = False


def tmap(fn, items, workers: int = WORKERS) -> list:
    items = list(items)
    if workers <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def _run(i: int):
    global _in_worker
    _in_worker = True
    fn, items = _job
    return fn(items[i])


def pmap(fn, items, workers: int = WORKERS) -> list:
    global _job
    items = list(items)
    if workers <= 1 or len(items) < 2 or _in_worker or sys.platform != "linux":
        return [fn(x) for x in items]
    _job = (fn, items)
    try:
        n = min(workers, len(items))
        with warnings.catch_warnings():
            # The only other threads at this point are NumPy's BLAS pool, which is fork-safe (pthread_atfork);
            # never call pmap while a Python thread pool (tmap, vlm.ask) is running.
            warnings.filterwarnings("ignore", "This process .* is multi-threaded", DeprecationWarning)
            with ProcessPoolExecutor(n, mp_context=mp.get_context("fork")) as ex:
                return list(ex.map(_run, range(len(items)), chunksize=max(1, len(items) // (n * 4))))
    finally:
        _job = None


if __name__ == "__main__":
    k = 3
    assert pmap(lambda x: x * k, range(50)) == tmap(lambda x: x * k, range(50)) == [x * k for x in range(50)]
    print("ok")
