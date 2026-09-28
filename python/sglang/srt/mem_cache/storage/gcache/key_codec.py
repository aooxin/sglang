"""Legacy GCache object names, independent of host-pool layout and SDK I/O."""

from dataclasses import dataclass

from sglang.srt.mem_cache.hicache_storage import PoolName


@dataclass(frozen=True)
class GCacheKeyCodec:
    model_name: str
    tp_rank: int
    draft_model_name: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("GCache requires a non-empty model_name")
        if not isinstance(self.tp_rank, int) or self.tp_rank < 0:
            raise ValueError("GCache requires a non-negative attention TP rank")
        if self.draft_model_name is not None and not self.draft_model_name.strip():
            raise ValueError("GCache draft_model_name must not be empty")

    def component_keys(self, pool_name: PoolName, page_hashes: list[str]) -> list[str]:
        if pool_name not in (
            PoolName.KV,
            PoolName.SWA,
            PoolName.DRAFT,
            PoolName.DRAFT_SWA,
        ):
            raise ValueError(f"Unsupported GCache pool: {pool_name}")

        prefix = self.model_name
        if pool_name in (PoolName.DRAFT, PoolName.DRAFT_SWA):
            prefix = self.draft_model_name or f"d:{self.model_name}"
        kind = "swa" if pool_name in (PoolName.SWA, PoolName.DRAFT_SWA) else "full"
        prefix += "_"
        suffix = f"_{self.tp_rank}_{kind}_"
        # Page-major K, V order matches MHA host-pool get_page_buffer_meta().
        return [
            prefix + page_hash + suffix + component
            for page_hash in page_hashes
            for component in ("k", "v")
        ]
