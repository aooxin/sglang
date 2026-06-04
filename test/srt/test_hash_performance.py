"""
Performance benchmark: SHA-256 vs xxhash64 for cache block hashing.

This test compares the performance of hashlib.sha256 (current implementation)
with xxhash.xxh64 (proposed optimization) in the context of SGLang's
KV cache page hashing (get_hash_str function).

Background:
-----------
In SGLang's HiCache / disaggregated serving, every page of tokens needs to be
hashed to create a unique key for cache lookup. The function `get_hash_str`
(defined in python/sglang/srt/mem_cache/hicache_storage.py) is called in a tight loop:

    for i in range(0, len(batch_tokens), self.page_size):
        last_hash = self.get_hash_str(
            batch_tokens[i : i + self.page_size], last_hash
        )

Tests cover:
  - Regular int tokens (page_size=1 and page_size=64)
  - EAGLE bigram tuple tokens (page_size=1 and page_size=64)
  - Sequence lengths: 1K, 2K, 4K, 8K, 16K, 32K

Usage:
------
    pip install xxhash
    python test/srt/test_hash_performance.py
"""

import hashlib
import random
import statistics
import struct
import time

import numpy as np

# Try to import xxhash
try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    print("WARNING: xxhash not installed. Run: pip install xxhash>=3.0.0")


NUM_WARMUP = 50  # Warmup iterations (discarded)
NUM_ROUNDS = 10  # Number of independent measurement rounds
NUM_ITERATIONS_PER_ROUND = (
    200  # Iterations within each round (averaged to get 1 sample)
)


# ============================================================================
# Implementation 1: SHA-256 (original) — per-token loop
# ============================================================================
def get_hash_str_sha256(token_ids, prior_hash=None):
    """Original implementation using hashlib.sha256 with per-token loop."""
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


# ============================================================================
# Implementation 2: SHA-256 (batch) — struct.pack optimization
# This is the approach applied to the project's get_hash_str in utils.py.
# ============================================================================
def get_hash_str_sha256_batch(token_ids, prior_hash=None):
    """SHA-256 with struct.pack batch optimization (current project implementation)."""
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    if len(token_ids) == 0:
        pass  # Empty list, no need to update hasher
    else:
        first = next(iter(token_ids))
        is_tuple_mode = isinstance(first, tuple)

        if is_tuple_mode:
            flattened = []
            for t in token_ids:
                flattened.extend(t)
            buf = struct.pack(f"<{len(flattened)}I", *flattened)
            hasher.update(buf)
        else:
            buf = struct.pack(f"<{len(token_ids)}I", *token_ids)
            hasher.update(buf)

    return hasher.hexdigest()


# ============================================================================
# Implementation 2b: SHA-256 (numpy) — np.array optimization
# This is the latest approach applied to the project's get_hash_str.
# Uses np.array(token_ids, dtype="<u4").tobytes() to convert both List[int]
# and List[tuple] uniformly without branching.
# ============================================================================
def get_hash_str_sha256_numpy(token_ids, prior_hash=None):
    """SHA-256 with numpy array optimization (latest project implementation)."""
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    if len(token_ids) > 0:
        # Works for both List[int] (1D) and List[tuple] (2D, e.g. EAGLE bigram mode).
        # np.array automatically handles both cases:
        #   - List[int] -> shape (n_tokens,)
        #   - List[tuple] -> shape (n_tokens, tuple_len)
        # .tobytes() produces C-contiguous little-endian uint32 bytes.
        arr = np.array(token_ids, dtype="<u4")
        hasher.update(arr.tobytes())

    return hasher.hexdigest()


# ============================================================================
# Implementation 3: xxhash64 (drop-in) — only replace hash algorithm
# ============================================================================
def get_hash_str_xxhash64(token_ids, prior_hash=None):
    """Drop-in replacement: only swap hashlib.sha256 -> xxhash.xxh64."""
    hasher = xxhash.xxh64()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


# ============================================================================
# Implementation 4: xxhash64 (batch) — struct.pack + xxhash64
# ============================================================================
def get_hash_str_xxhash64_batch(token_ids, prior_hash=None):
    """xxhash64 with struct.pack batch optimization."""
    hasher = xxhash.xxh64()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    if len(token_ids) == 0:
        pass  # Empty list, no need to update hasher
    else:
        first = next(iter(token_ids))
        is_tuple_mode = isinstance(first, tuple)

        if is_tuple_mode:
            flattened = []
            for t in token_ids:
                flattened.extend(t)
            buf = struct.pack(f"<{len(flattened)}I", *flattened)
            hasher.update(buf)
        else:
            buf = struct.pack(f"<{len(token_ids)}I", *token_ids)
            hasher.update(buf)

    return hasher.hexdigest()


# ============================================================================
# Implementation 5: xxhash64 (oneshot) — struct.pack + single-shot hash
# ============================================================================
def get_hash_str_xxhash64_oneshot(token_ids, prior_hash=None):
    """Most optimized: build full byte buffer, then hash in one shot."""
    buf = b""

    if len(token_ids) == 0:
        pass  # Empty list, no token bytes to pack
    else:
        first = next(iter(token_ids))
        is_tuple_mode = isinstance(first, tuple)

        if is_tuple_mode:
            flattened = []
            for t in token_ids:
                flattened.extend(t)
            buf = struct.pack(f"<{len(flattened)}I", *flattened)
        else:
            buf = struct.pack(f"<{len(token_ids)}I", *token_ids)

    if prior_hash:
        buf = bytes.fromhex(prior_hash) + buf

    return xxhash.xxh64(buf).hexdigest()


# ============================================================================
# Benchmark helper
# ============================================================================
def benchmark_one_sequence(
    func,
    pages,
    num_warmup=NUM_WARMUP,
    num_rounds=NUM_ROUNDS,
    num_iterations_per_round=NUM_ITERATIONS_PER_ROUND,
):
    """
    Run multiple independent rounds, each averaging many iterations.
    Returns: (mean_ms, std_ms) across rounds.
    """
    # Warmup
    for _ in range(num_warmup):
        last_hash = None
        for page in pages:
            last_hash = func(page, last_hash)

    # Measurement
    round_averages = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        for _ in range(num_iterations_per_round):
            last_hash = None
            for page in pages:
                last_hash = func(page, last_hash)
        t1 = time.perf_counter()
        avg_per_seq = (t1 - t0) * 1000 / num_iterations_per_round
        round_averages.append(avg_per_seq)

    return statistics.mean(round_averages), statistics.stdev(round_averages)


def generate_int_pages(num_tokens, page_size, vocab_size=32000):
    """Generate random int token pages."""
    tokens = [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
    return [tokens[i : i + page_size] for i in range(0, num_tokens, page_size)]


def generate_tuple_pages(num_tokens, page_size, vocab_size=32000):
    """Generate random EAGLE bigram (tuple) token pages."""
    tuples = [
        (random.randint(0, vocab_size - 1), random.randint(0, vocab_size - 1))
        for _ in range(num_tokens)
    ]
    return [tuples[i : i + page_size] for i in range(0, num_tokens, page_size)]


# ============================================================================
# Correctness test
# ============================================================================
def test_hash_consistency():
    """Verify all batch implementations produce identical output to original."""
    print("=" * 80)
    print("  TEST: Hash Consistency — original vs batch vs numpy (int tokens)")
    print("=" * 80)
    print()

    test_cases = [
        ("empty list", [], None),
        ("empty with prior", [], "ab" * 32),
        ("1 int", [12345], None),
        ("8 ints, no prior", [1, 2, 3, 4, 5, 6, 7, 8], None),
        ("8 ints, with prior", [1, 2, 3, 4, 5, 6, 7, 8], "deadbeef" * 8),
        ("64 ints (1 page)", list(range(64)), None),
        ("64 ints, with prior", list(range(64)), "00" * 32),
        ("all zeros 64", [0] * 64, None),
        ("large values", [2**31 - 1] * 16, None),
        ("random 64", [random.randint(0, 32000) for _ in range(64)], None),
        ("random 256", [random.randint(0, 32000) for _ in range(256)], None),
        ("chained prior", [100, 200, 300], get_hash_str_sha256([10, 20, 30])),
    ]

    print(
        f"  {'#':<4} {'Description':<22} {'Orig==Batch':<13} {'Orig==Numpy':<13} {'All Match'}"
    )
    print(f"  {'─'*4} {'─'*22} {'─'*12} {'─'*12} {'─'*9}")

    all_match = True
    for idx, (desc, tokens, prior) in enumerate(test_cases, 1):
        h_orig = get_hash_str_sha256(tokens, prior)
        h_batch = get_hash_str_sha256_batch(tokens, prior)
        h_numpy = get_hash_str_sha256_numpy(tokens, prior)
        match_batch = h_orig == h_batch
        match_numpy = h_orig == h_numpy
        match_all = match_batch and match_numpy
        if not match_all:
            all_match = False
        print(
            f"  {idx:<4} {desc:<22} {'YES' if match_batch else 'FAIL':<13} "
            f"{'YES' if match_numpy else 'FAIL':<13} {'YES' if match_all else 'FAIL'}"
        )

    # Chaining test (32 pages)
    print()
    print("  --- Chaining test: 2048 tokens, page_size=64, 32 pages ---")
    tokens = [random.randint(0, 32000) for _ in range(2048)]
    pages = [tokens[i : i + 64] for i in range(0, 2048, 64)]
    last_orig = None
    last_batch = None
    last_numpy = None
    for page in pages:
        last_orig = get_hash_str_sha256(page, last_orig)
        last_batch = get_hash_str_sha256_batch(page, last_batch)
        last_numpy = get_hash_str_sha256_numpy(page, last_numpy)
        if last_orig != last_batch or last_orig != last_numpy:
            all_match = False
    chain_match = last_orig == last_batch == last_numpy
    print(f"  32 pages chained: {'MATCH' if chain_match else 'FAIL'}")

    print()
    print("=" * 80)
    print("  TEST: Hash Consistency — original vs batch vs numpy (EAGLE tuple tokens)")
    print("=" * 80)
    print()

    tuple_cases = [
        ("1 tuple", [(1, 2)], None),
        ("4 tuples, no prior", [(1, 2), (3, 4), (5, 6), (7, 8)], None),
        ("4 tuples, with prior", [(1, 2), (3, 4), (5, 6), (7, 8)], "ab" * 32),
        ("64 tuples (1 page)", [(i, i + 1) for i in range(64)], None),
        ("64 tuples, with prior", [(i, i + 1) for i in range(64)], "ff" * 32),
        (
            "random 64 tuples",
            [(random.randint(0, 32000), random.randint(0, 32000)) for _ in range(64)],
            None,
        ),
        (
            "random 64, with prior",
            [(random.randint(0, 32000), random.randint(0, 32000)) for _ in range(64)],
            "deadbeef" * 8,
        ),
        (
            "3-tuples",
            [(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)],
            None,
        ),
        (
            "3-tuples, with prior",
            [(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)],
            "cafe" * 16,
        ),
    ]

    print(
        f"  {'#':<4} {'Description':<22} {'Orig==Batch':<13} {'Orig==Numpy':<13} {'All Match'}"
    )
    print(f"  {'─'*4} {'─'*22} {'─'*12} {'─'*12} {'─'*9}")

    for idx, (desc, tokens, prior) in enumerate(tuple_cases, 1):
        h_orig = get_hash_str_sha256(tokens, prior)
        h_batch = get_hash_str_sha256_batch(tokens, prior)
        h_numpy = get_hash_str_sha256_numpy(tokens, prior)
        match_batch = h_orig == h_batch
        match_numpy = h_orig == h_numpy
        match_all = match_batch and match_numpy
        if not match_all:
            all_match = False
        print(
            f"  {idx:<4} {desc:<22} {'YES' if match_batch else 'FAIL':<13} "
            f"{'YES' if match_numpy else 'FAIL':<13} {'YES' if match_all else 'FAIL'}"
        )

    # Tuple chaining test (8 pages of 64 tuples)
    print()
    print("  --- Chaining test: 512 tuples, page_size=64, 8 pages ---")
    tuples = [(random.randint(0, 32000), random.randint(0, 32000)) for _ in range(512)]
    pages = [tuples[i : i + 64] for i in range(0, 512, 64)]
    last_orig = None
    last_batch = None
    last_numpy = None
    for page in pages:
        last_orig = get_hash_str_sha256(page, last_orig)
        last_batch = get_hash_str_sha256_batch(page, last_batch)
        last_numpy = get_hash_str_sha256_numpy(page, last_numpy)
        if last_orig != last_batch or last_orig != last_numpy:
            all_match = False
    chain_match = last_orig == last_batch == last_numpy
    print(f"  8 pages chained: {'MATCH' if chain_match else 'FAIL'}")

    print()
    print("  " + "=" * 76)
    if all_match:
        print(
            "  RESULT: ALL PASSED — batch and numpy optimizations produce identical hashes."
        )
    else:
        print("  RESULT: MISMATCH DETECTED!")
    print("  " + "=" * 76)
    print()
    return all_match


# ============================================================================
# Performance benchmark
# ============================================================================
def run_benchmark_section(
    title, page_size, seq_lengths, page_generator, implementations
):
    """Run benchmark for a specific page_size and token type."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(
        f"  page_size={page_size}, {NUM_ROUNDS} rounds x {NUM_ITERATIONS_PER_ROUND} iters, {NUM_WARMUP} warmup"
    )
    print()

    results = {name: {} for name, _ in implementations}

    for num_tokens in seq_lengths:
        pages = page_generator(num_tokens, page_size)
        num_pages = len(pages)

        print("-" * 80)
        print(
            f"  Sequence: {num_tokens} tokens ({num_pages} pages x {page_size} tokens/page)"
        )
        print("-" * 80)
        print(
            f"  {'Implementation':<22} {'Time (ms)':<20} {'Speedup':<14} {'Pages/s':<12}"
        )
        print(f"  {'─'*22} {'─'*19} {'─'*13} {'─'*11}")

        baseline_avg = None
        baseline_std = None
        for name, func in implementations:
            avg, std = benchmark_one_sequence(func, pages)

            if baseline_avg is None:
                baseline_avg = avg
                baseline_std = std

            speedup = baseline_avg / avg if avg > 0 else float("inf")
            rel_base = baseline_std / baseline_avg if baseline_avg > 0 else 0
            rel_cur = std / avg if avg > 0 else 0
            speedup_std = speedup * (rel_base**2 + rel_cur**2) ** 0.5

            pages_per_sec = num_pages / (avg / 1000) if avg > 0 else 0

            time_str = f"{avg:.4f} +/- {std:.4f}"
            speedup_str = f"{speedup:.2f}x +/- {speedup_std:.2f}"
            print(
                f"  {name:<22} {time_str:<20} {speedup_str:<14} {pages_per_sec:<12.0f}"
            )

            results[name][num_tokens] = (avg, std, speedup, speedup_std)

        print()

    # Summary table
    print("-" * 80)
    print(f"  SUMMARY: Speedup [mean +/- std]")
    print("-" * 80)
    col_width = 14
    header = f"  {'Implementation':<22}"
    for sl in seq_lengths:
        header += f"{'%dK' % (sl // 1024):>{col_width}}"
    print(header)
    print(f"  {'─'*22}" + f"{'─'*col_width}" * len(seq_lengths))
    for name, _ in implementations:
        row = f"  {name:<22}"
        for sl in seq_lengths:
            _, _, spd, spd_std = results[name][sl]
            row += f"{spd:.2f}x+/-{spd_std:.2f}".rjust(col_width)
        print(row)
    print()


def run_benchmark():
    if not HAS_XXHASH:
        print("ERROR: xxhash not installed. Run: pip install xxhash>=3.0.0")
        return

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    # All implementations with tuple support
    implementations = [
        ("SHA-256 (original)", get_hash_str_sha256),
        ("SHA-256 (batch)", get_hash_str_sha256_batch),
        ("SHA-256 (numpy)", get_hash_str_sha256_numpy),
        ("xxhash64 (drop-in)", get_hash_str_xxhash64),
        ("xxhash64 (batch)", get_hash_str_xxhash64_batch),
        ("xxhash64 (oneshot)", get_hash_str_xxhash64_oneshot),
    ]

    # --- Test 1: Regular int tokens, page_size=64 ---
    run_benchmark_section(
        "Benchmark: INT tokens, page_size=64",
        page_size=64,
        seq_lengths=seq_lengths,
        page_generator=generate_int_pages,
        implementations=implementations,
    )

    # --- Test 2: Regular int tokens, page_size=1 ---
    run_benchmark_section(
        "Benchmark: INT tokens, page_size=1",
        page_size=1,
        seq_lengths=[1024, 2048, 4096],  # smaller set for page_size=1 (very many pages)
        page_generator=generate_int_pages,
        implementations=implementations,
    )

    # --- Test 3: EAGLE tuple tokens, page_size=64 ---
    run_benchmark_section(
        "Benchmark: EAGLE TUPLE tokens, page_size=64",
        page_size=64,
        seq_lengths=seq_lengths,
        page_generator=generate_tuple_pages,
        implementations=implementations,
    )

    # --- Test 4: EAGLE tuple tokens, page_size=1 ---
    run_benchmark_section(
        "Benchmark: EAGLE TUPLE tokens, page_size=1",
        page_size=1,
        seq_lengths=[1024, 2048, 4096],
        page_generator=generate_tuple_pages,
        implementations=implementations,
    )


if __name__ == "__main__":
    random.seed(42)
    test_hash_consistency()
    run_benchmark()
