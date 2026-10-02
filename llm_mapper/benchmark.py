"""Synchronized mapping benchmarks with sampled process/device memory."""
import threading
import time
import platform
import psutil
from .devices import synchronize
from .index import map_records
from .sequences import SequenceReader


class MemorySampler:
    """RSS and MPS driver allocations overlap on unified memory: never sum them."""
    def __init__(self, device, interval=0.01):
        self.device, self.interval = device, interval
        self.peak_rss = self.peak_device = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        import torch
        process = psutil.Process()
        while True:
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            if self.device == "mps":
                self.peak_device = max(self.peak_device, torch.mps.driver_allocated_memory())
            elif self.device == "cuda":
                self.peak_device = max(self.peak_device, torch.cuda.max_memory_allocated())
            if self.stop.wait(self.interval):
                return

    def __enter__(self):
        if self.device == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def benchmark(path, index, batch_size=8, truth=None, reference=None):
    """truth maps read ID to gene/reference label; metrics are per emitted chunk.

    Model loading and one warmup batch are excluded. Missing truth is an error,
    preventing an accidentally optimistic accuracy on just the labeled subset.
    """
    from itertools import islice
    warmup = list(islice(map_records(SequenceReader(path, reference=reference), index, batch_size), batch_size))
    if not warmup:
        raise ValueError("No reads to benchmark")
    synchronize(index.embedder.device)
    count = correct = bases = 0
    with MemorySampler(index.embedder.device) as memory:
        started = time.perf_counter()
        for prediction in map_records(SequenceReader(path, reference=reference), index, batch_size):
            count += 1
            bases += prediction["read_end"] - prediction["read_start"]
            if truth is not None:
                expected = truth[prediction["read_id"]]
                correct += prediction["predicted_label"] == expected
        synchronize(index.embedder.device)
        elapsed = time.perf_counter() - started
    return {"device": index.embedder.device, "platform": platform.platform(),
            "fingerprint": index.embedder.fingerprint, "batch_size": batch_size,
            "chunks": count, "seconds": elapsed, "chunks_per_second": count / elapsed,
            "bases_per_second": bases / elapsed,
            "accuracy_per_chunk": correct / count if truth is not None else None,
            "sampled_peak_rss_bytes": memory.peak_rss,
            "sampled_peak_device_bytes": memory.peak_device,
            "memory_sampling_interval_seconds": memory.interval,
            "memory_note": "RSS and device memory overlap on Apple Silicon; do not sum. "
                           "Sampled maxima can miss transient peaks; model loading excluded."}
