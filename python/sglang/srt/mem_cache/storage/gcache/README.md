# GCache storage for UnifiedRadixCache

GCache implements MHA KV, SWA, and draft storage beneath the 0520
`UnifiedRadixCache`, `HostPoolGroup`, and `HybridCacheController`. Host allocation,
page indexing, eviction, and device/host transfers remain owned by HiCache.

## Support and validation

| Area | Support |
| --- | --- |
| Primary KV | `batch_exists`, `batch_get_v1`, `batch_set_v1`; also accepted by v2 |
| Side pools | SWA, DRAFT, DRAFT_SWA via named registration and v2 exists/get/set |
| Host pools | `MHATokenToKVPoolHost` and `AsymmetricMHATokenToKVPoolHost` |
| Layouts | `page_first` with kernel I/O; `page_first_direct` with direct I/O |
| Parallelism | Homogeneous attention TP, including DP attention; PP=CP=DCP=1 |
| Draft assembly | Ordinary MHA draft; all-SWA draft with the existing target SWA pool/allocator |
| Host memory mode | CPU integration covers `cache`; `buffer_only` has not been validated |
| Rejected formats | MLA, Mamba, indexers/DSA, DeepSeek V4, packed MTP draft layers, empty physical pools |
| Rejected topology | PP/CP/DCP > 1, heterogeneous TP, head splitting |
| Local CPU validation | Tensor bytes over a subprocess mockserver; two Gloo ranks; actual assembler/controller storage threads |
| Local GPU validation | H20-3e CUDA device/host/server round trips; both layouts, FP16/BF16, symmetric/asymmetric KV, ordinary/all-SWA draft |
| Deployment validation | Native SDK import checked; real GCache/RDMA, GPU model inference, performance, and image build still need a deployment environment |

Each pool uses its own capacity, page size, K/V component lengths, registered
allocations, and host indices. The backend consumes the Full/SWA mappings supplied
by HiCache. Re-registering the same pool and underlying allocation is idempotent.
An all-SWA draft registers only DRAFT_SWA and never requires Full draft objects.
The existing assembler does not establish support for arbitrary draft models
that simultaneously require Full and SWA sidecars; those need separate validation.

## SDK and image

The optional SDK version is `gcache-sdk==2.1.0+g5978d0b2.py312.release.2.1`.
The [pinned wheel](https://cnbj1-fds.api.xiaomi.net/juicefs/gcache/release/2.1/20260917-5978d0b2/gcache_sdk-2.1.0+g5978d0b2.py312.release.2.1-py3-none-any.whl)
contains `gcache_py_bind.cpython-312-x86_64-linux-gnu.so` despite its
`py3-none-any` filename. It requires CPython 3.12 on Linux x86_64. Its SHA256 is:

```text
33df5f536cf06b16fce597b1461e9a52755c0d29ef584850f68dc1e94a6daf8b
```

`docker/Dockerfile.xiaomi` inherits `lmsysorg/sglang:v0.5.20`, checks the base's
version and editable source path, installs the wheel with its checksum, and
overlays this checkout. `BASE_IMAGE` may select an internal mirror of the same
release. Build from the repository root on the packaging machine:

```bash
docker build -f docker/Dockerfile.xiaomi -t sglang:0.5.20-gcache .
```

The image retains the base's dependencies and compiled extensions. Changes to
dependencies or compiled source require a compatible rebuilt base. The image
also copies `docker/gcache.flags` and installs
`/home/work/gcache/wait-gcache-server.sh`. The script checks master discovery
only and is not run automatically; it does not prove RDMA readiness.

The SDK import is lazy. Services using other storage backends do not need it.

## Configuration

| Environment variable | Meaning |
| --- | --- |
| `GCACHE_MASTER_ADDR` | Required master address, without an HTTP scheme |
| `GCACHE_CACHE_GROUP` | Required cache group |
| `HPN_IB_IFNAME` | Preferred client NIC list |
| `GCACHE_CLIENT_IFNAME` | Legacy fallback; a single NIC explicitly overrides the HPN list |
| `HPN_IB_HCA` | Optional `rdma_allowed_devices` override, appended even if absent from the template |
| `GCACHE_CLIENT_THREADS` | Positive integer; default 8 |
| `GCACHE_TRAFFIC_CLASS` | Integer from -1 through 255; default -1 |
| `GCACHE_CONFIG_FILE` | Optional SDK flags template |
| `SGLANG_HICACHE_DRAFT_MODEL_NAME` | Optional draft namespace; default `d:<served_model_name>` |

Add these server options to the model's normal launch configuration:

```text
--enable-hierarchical-cache
--hicache-storage-backend gcache
--hicache-mem-layout page_first
--hicache-io-backend kernel
```

`layer_first` is resolved to a supported page layout for kernel/direct I/O.
Unsupported startup layouts and I/O backends fail in argument validation;
pool types and topology are checked again at backend initialization/registration.

`--hicache-storage-backend-extra-config` accepts these backend-specific fields:

```json
{
  "config_file": "/path/to/template.flags",
  "config_dir": "/tmp/gcache-config",
  "log_dir": "/tmp/gcache",
  "rpc_socket_type": "rdma",
  "timeout_us": 0
}
```

All fields are optional. Template selection is explicit `config_file`, then
`GCACHE_CONFIG_FILE`, then `/sgl-workspace/gcache.flags` if present, then built-in
defaults matching the legacy image. Overrides replace existing flags and append
missing ones. Generated files have unique DP/TP-qualified names and are reused
by the same process client across reattachment. Without path overrides, flags
and logs use a short temporary directory `gcache-<uid>`; the SDK places a Unix
socket under the log directory, so custom log paths must stay short.
`rpc_socket_type="tcp"` is forwarded to the SDK but has not been validated as an
alternative to RDMA for the supplied service.

## Object names and failure contracts

Each logical page contains two GCache objects, in K then V order. For a page
hash `h` and attention TP rank `r`:

| Pool | Keys |
| --- | --- |
| KV | `<target>_h_r_full_k`, `<target>_h_r_full_v` |
| SWA | `<target>_h_r_swa_k`, `<target>_h_r_swa_v` |
| DRAFT | `<draft>_h_r_full_k`, `<draft>_h_r_full_v` |
| DRAFT_SWA | `<draft>_h_r_swa_k`, `<draft>_h_r_swa_v` |

`<target>` is the served model name. `<draft>` is
`SGLANG_HICACHE_DRAFT_MODEL_NAME` when set, otherwise `d:<target>`.
DP rank belongs in process configuration and metrics, not object keys.

K/V addresses and byte lengths come from each host pool's
`get_page_buffer_meta()`. Asymmetric K/V allocations are registered separately;
views sharing storage are deduplicated by their actual allocation. Pointers must
lie within allocations registered for that specific pool. Invalid, duplicate,
unaligned, out-of-range, or incomplete page indices fail before native I/O.

Both components must succeed for a page to succeed. Writes first query with
`for_read=False` and explicit sizes, then submit only missing objects. Reads and
prefetch queries use read semantics. Empty batches perform no SDK I/O.
Exceptions, malformed result lists, and invalid metadata return explicit failed
pages. Initialization and registration errors propagate to the caller.

`batch_exists_v2()` intersects each pool's valid prefix endpoints:

- Primary KV must have a contiguous prefix of complete K/V pages.
- `ALL_PAGES` requires every page up to the endpoint.
- `TRAILING_PAGES` requires only the window immediately before the endpoint.
  Window length comes from the transfer's key count; actual queries always use
  the primary hashes, since prefetch transfers may contain placeholder keys.

For six Full pages, SWA hits `[1, 1, 0, 1, 1, 1]`, and a two-page window,
`restorable_prefix_pages` is `[1, 2, 5, 6]`. With another rank whose valid
endpoints are `[1, 2, 3, 4]`, the controller restores two pages. It intersects
endpoint bitmaps for this backend instead of taking the minimum of the maxima
(four pages would be unsafe). Cancellation contributes only the zero endpoint.
The existing completion and release paths handle objects lost between query and
read, including per-rank Full/SWA failures.

## Lifetime and metrics

The pinned SDK is a process singleton. Its `close()` calls global
`gcache_fini()`, and it provides neither a singleton reset nor memory unregister.
`GCacheClientRuntime` owns the SDK client, flags file, and registrations
for the worker's lifetime:

- Backend `close()` rejects new work and waits for in-flight operations.
  Detaching a controller never calls the SDK's global `close()`.
- Reattaching with the same settings reuses the client, flags, and registrations.
  New pools register their own allocations. A failed side-pool registration
  unwinds controller attachment; a failed dynamic registration is not published.
- Registered allocations, including allocations whose native registration may
  have partially failed, remain alive until worker exit. Repeatedly attaching
  entirely new host allocations can therefore grow retained memory. A worker
  restart is required to reclaim it fully.
- Changing rendered SDK flags or timeout in the same process raises an error.
  Restart the worker to change connection settings. Initialize GCache after
  creating worker processes, not before forking them.

Standard storage metrics report completed pages and transferred GiB/s for
prefetch and backup. Existing objects count as successful backup pages but do
not contribute transferred bytes. With storage metrics enabled, these
Prometheus series are exported for both `prefetch` and `backup`, labelled by
`storage_backend`, `tp_rank`, and `dp_rank`:

| Metric suffix, after `sglang:gcache_<direction>_` | Meaning |
| --- | --- |
| `bytes_total` | Bytes successfully transferred, excluding skipped existing writes |
| `preprocess_seconds` | Metadata and pointer validation time |
| `rdma_seconds` | SDK call time, including network/server work and write existence queries |
| `postprocess_seconds` | Object-to-page result processing time |
| `idle_seconds` | Gap between consecutive calls in this direction |
| `rdma_bandwidth` | Successful bytes divided by SDK duration, in GiB/s |
| `batch_keys` | Logical pages per pool per batch; each page has two objects |

Timing, bandwidth, and page series are histograms. Collectors are reused across
reattachment. The historical `rdma` name refers to SDK call duration; it does not
itself establish that RDMA was used.

## Tests

The CPU tests use a real, independent `GCacheMockServer` process, backed by an
in-memory KV dictionary. `NetworkGCacheClient` implements the SDK's pointer/size
API and sends actual tensor bytes over loopback HTTP/JSON. The client holds no KV
dictionary. Different clients and ranks exchange bytes through the server;
stopping the server causes reads to fail. Per-key errors, timeouts, malformed
statuses, held requests, and registration failures are injected on the server.

```text
Controller -> GcacheStorage -> NetworkGCacheClient -> HTTP -> mockserver KV
```

Only the external SDK entry and CPU fixture setup are substituted. Storage I/O
is not replaced by function return values. The transport is a test protocol,
not the native GCache byte-RPC/RDMA wire protocol. Production always uses the
official SDK and has no mockserver fallback.

`test/registered/unit/mem_cache/fixtures/gcache_legacy_contract.json` records
legacy key/hash vectors and page byte digests captured from the source MHA
allocation/metadata methods. Fixtures cover FP16, symmetric/asymmetric K/V,
both layouts, and disjoint pages. Tests also copy FP16/BF16/uint8 tensor bytes
through the server. They exercise independent Full/SWA pool geometry, draft
namespaces, the real pool assembler, storage worker threads, runtime attach,
and two actual Gloo processes. GPU events/allocation setup are substituted for
the CPU controller fixtures; no GPU transfer kernels run in this suite.

Run from the repository root in an environment with the 0520 Python dependencies:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_gcache_*.py
```

The GPU suite at
`test/registered/e2e/hicache/test_hicache_storage_gcache_gpu.py` uses the real
`MHATokenToKVPool`/host constructors, `cudaHostRegister`, `L2TransferEngine`,
CUDA streams/events, staged write-back kernels, and direct transfers. It writes
device data into the mockserver, erases the writer's GPU and host buffers, then
restores into separate host/device pools at different page indices. Assertions
compare the server objects and restored device bytes with the original device
values, and check that untouched device pages remain zero. No CUDA allocation,
copy, stream, event, or kernel is mocked.

```bash
CUDA_HOME=/usr/local/cuda PYTHONPATH=python python -m pytest -q \
  test/registered/e2e/hicache/test_hicache_storage_gcache_gpu.py
```

The 16 GPU cases cover both layouts, FP16/BF16, symmetric/asymmetric primary
KV, and ordinary/all-SWA draft. They have passed on H20-3e (SM90) with Python
3.12, PyTorch 2.8.0+cu128, and CUDA 13.1 for JIT compilation. This validates
the physical GPU/host storage path on that environment; it does not launch a
model or certify the official image's complete dependency stack. CPU CI skips
this suite; it is registered separately for GPU CI.

The mockserver can also run by itself, using only the Python standard library:

```bash
python python/sglang/test/gcache_mock_server.py --port 18080
```

The opt-in native SDK test at
`test/registered/e2e/hicache/test_hicache_storage_gcache_backend.py` covers all
four pools and both layouts, with symmetric and asymmetric K/V. It uses a
unique model/draft namespace and deletes only its own objects. In an environment
with the pinned SDK, a dedicated test cache group, and the required NIC settings:

```bash
SGLANG_TEST_GCACHE=1 PYTHONPATH=python python -m pytest -q \
  test/registered/e2e/hicache/test_hicache_storage_gcache_backend.py
```

These cases skip by default and are disabled in ordinary CI. Transport defaults
to RDMA; `GCACHE_TEST_RPC_SOCKET_TYPE` can select TCP for a compatible service.
The native test checks host-memory byte round trips, not model output or GPU
kernels. Real-service multi-rank faults, cold/warm L3 model output, TTFT,
bandwidth, CPU overhead, and the full image build require separate validation
on the target hardware. Local mockserver timing is not performance evidence.

## Cache compatibility and rollback

Reusing old objects requires identical model identity, token hash chain,
TP shard, page size, dtype, layer order, and physical encoding. Equal names or
byte lengths alone do not establish compatibility. The CPU vectors do not
validate model-specific FP8/FP4 encodings. Use a new served-model namespace or
cache group when these compatibility conditions have not been established.

For rollout, keep the previous image and launch configuration available. Roll
back by restarting workers with that image/configuration. Validation should use
an independent namespace/cache group; rollback does not require deleting shared
cache data.
