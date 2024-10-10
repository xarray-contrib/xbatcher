import time
from typing import List, Union, Tuple

import numpy as np
import dask
from distributed import get_client, Future, as_completed, Client
import dask.array as da
import xarray as xr


class BatchIDX:
    """Deterministic batch idx with [] syntax dropping remainder."""

    def __init__(self, array, batch_size: int):
        self.batch_size = batch_size
        self.array = array

    def __getitem__(self, i: int):
        start = i * self.batch_size
        end = start + self.batch_size
        return slice(start, end)

    def __len__(self):
        return int(np.floor(len(self.array) / self.batch_size))


class PrefetchBatchGenerator:
    """
    Using deterministic index generation over arrays that are chunked only over
    the first dimension, submit tasks to workers to load a batch per task and
    accumulate prefetch number of these futures. Block until the current buffer
    of futures is available (calling gather on a batch of batches), while a
    separate prefetch buffer of tasks has already been submitted.

    The idea is to submit tasks to dask in order to start loading that batch
    data into memory while actively transfering ready batches to the process
    where this class has been instantiated.

    Considerations:
    - does getting results as they complete slows things down compared to gather?
        - use .batches()! Still, is this the right approach?
    - Can this be pickled (if necessary) to be used in tf.data.Dataset.from_generator()?
        - does it have to be pickled? Can I set num_processes to 0 or 1 to avoid the
        need to pickle it if it does cause issues?
    - Does get_client slow things down?
    - How to set prefetch?
    - should we warn that 2xprefetch amount of data is too large for relevant workers?
    - should we make it a high priority to support heterogeneous worker types?
        - we might want to isolate worker types between data and modeling e.g.
        use gpu for modeling, but cpu workers for prefetching.
    """

    def __init__(self, array: da.Array, batch_size: int, prefetch: int):
        if not array.chunksize[1:] == array.shape[1:]:
            raise ValueError(
                "we require chunking only over the first, single, major, batch axis"
            )
        self.array = array
        self.batch_size = batch_size
        self._prefetch = prefetch
        # use register_worker_plugin here instead
        self._gen_idx = BatchIDX(array=array, batch_size=batch_size)
        self.total_batches = len(self._gen_idx)
        self._future_buffer: List[Future] = []
        self._prefetch_buffer: List[Future] = []
        self._init_prefetch = True
        self._idx_to_be_submitted = 0

    @property
    def prefetch_range(self) -> Tuple[int, int]:
        """Return the number of elements to submit."""
        remaining = self.total_batches - self._idx_to_be_submitted

        prefetch_count = min(remaining, self._prefetch)
        return self._idx_to_be_submitted, self._idx_to_be_submitted + prefetch_count

    def _submit_prefetch_batches(self):
        """..."""
        client = get_client()
        if self._init_prefetch:
            for i in range(*self.prefetch_range):
                idx_slice = self._gen_idx[i]
                future = client.submit(
                    _make_batch, array=self.array, idx=idx_slice, pure=False
                )
                self._future_buffer.append(future)
                self._idx_to_be_submitted = i + 1
            self._init_prefetch = False

        for i in range(*self.prefetch_range):
            idx_slice = self._gen_idx[i]
            future = client.submit(
                _make_batch, array=self.array, idx=idx_slice, pure=False
            )
            self._prefetch_buffer.append(future)
            self._idx_to_be_submitted = i + 1

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        """..."""
        client = get_client()
        n_submissions = max(int(np.ceil(self.total_batches / self._prefetch)), 1)
        for _ in range(n_submissions):
            self._submit_prefetch_batches()
            for batch_of_batches_futures in as_completed(self._future_buffer).batches():
                batch_of_batches = client.gather(batch_of_batches_futures)
                for batch in batch_of_batches:
                    yield batch
            self._future_buffer = self._prefetch_buffer
            self._prefetch_buffer = []


def _make_batch(
    array: Union[da.Array, xr.DataArray], idx: slice, scheduler: str = "single-threaded"
) -> np.ndarray:
    """..."""

    with dask.config.set(scheduler=scheduler):
        batch = np.asarray(array[idx])
    return batch


if __name__ == "__main__":

    import os

    # set env or edit to suit your needs!
    address = os.getenv("DASK_ADDRESS")
    remote_path = os.getenv("REMOTE_PATH")
    client = Client(address)

    # write out an example zarr dataset
    target_array = da.random.random((50000, 9, 100, 100), chunks=[1024, 9, 100, 100])
    target_array.to_zarr(remote_path, overwrite=True)

    def do_ml():
        # load array for your data problem
        array = da.from_zarr(remote_path)
        batch_gen = PrefetchBatchGenerator(
            array=array[:50000], batch_size=1024, prefetch=25
        )
        nbytes = 0
        shapes = []
        t0 = time.time()
        for b in batch_gen:
            nbytes += b.nbytes
            shapes.append(b.shape)
        elapsed = time.time() - t0
        gbps = (nbytes / elapsed) / 1024**3
        n = sum([s[0] for s in shapes])

        return gbps, n

    f = client.submit(do_ml, pure=False)
    gbps, n = f.result()
    print(f"We achieved about {gbps} gbps over {n} total examples")
