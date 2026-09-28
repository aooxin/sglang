"""Parse legacy GCache environment settings without mutating the environment."""

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Match the legacy image defaults when no external flags template is installed.
_DEFAULT_FLAGS = """--use_byterpc=true
--log_level=1
--log_filename_suffix_use_pid=true
--verify_checksum=false
--calculate_checksum=false
--set_max_open_fds=true
--force_same_rail_nic=true
--fixed_numa_node=false
--byterpc_byte_express_memory_pool_max_regions=50
"""


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{name} must be a single-line string")
    return value.strip()


def _integer(name: str, value, minimum: int, maximum: int | None = None) -> int:
    try:
        if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value).strip()):
            raise ValueError
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return result


def render_flags(template: str, overrides: dict) -> str:
    """Replace existing flags and append absent flags (notably HPN_IB_HCA)."""
    values = {}
    for key, value in overrides.items():
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", key):
            raise ValueError(f"Invalid GCache flag name: {key!r}")
        value = str(value).lower() if isinstance(value, bool) else str(value)
        values[key] = _text(key, value)

    lines = []
    seen = set()
    for line in template.splitlines():
        match = re.match(r"\s*--([a-zA-Z_][a-zA-Z_0-9]*)\s*=", line)
        key = match[1] if match else None
        if key in values:
            lines.append(f"--{key}={values[key]}")
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(f"--{key}={value}" for key, value in values.items() if key not in seen)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class GCacheClientConfig:
    master_addr: str
    cache_group: str
    client_ifname: str
    client_threads: int
    traffic_class: int
    rdma_allowed_devices: str
    rpc_socket_type: str
    timeout_us: int
    config_file: Path | None
    config_dir: Path
    log_dir: Path

    @classmethod
    def from_env(cls, extra_config: dict | None = None) -> "GCacheClientConfig":
        extra = dict(extra_config or {})
        unknown = extra.keys() - {
            "config_file",
            "config_dir",
            "log_dir",
            "rpc_socket_type",
            "timeout_us",
        }
        if unknown:
            raise ValueError(f"Unknown GCache extra config fields: {sorted(unknown)}")

        master_addr = _text(
            "GCACHE_MASTER_ADDR", os.environ.get("GCACHE_MASTER_ADDR", "")
        )
        cache_group = _text(
            "GCACHE_CACHE_GROUP", os.environ.get("GCACHE_CACHE_GROUP", "")
        )
        for name, value in (
            ("GCACHE_MASTER_ADDR", master_addr),
            ("GCACHE_CACHE_GROUP", cache_group),
        ):
            if not value:
                raise ValueError(f"GCache requires {name}")

        hpn = _text("HPN_IB_IFNAME", os.environ.get("HPN_IB_IFNAME", ""))
        legacy = _text(
            "GCACHE_CLIENT_IFNAME", os.environ.get("GCACHE_CLIENT_IFNAME", "")
        )
        client_ifname = hpn or legacy
        if legacy and len(legacy.split(",")) == 1:
            client_ifname = legacy
        if not client_ifname:
            raise ValueError("GCache requires HPN_IB_IFNAME or GCACHE_CLIENT_IFNAME")

        transport = _text("rpc_socket_type", extra.get("rpc_socket_type", "rdma"))
        if transport not in ("rdma", "tcp"):
            raise ValueError("GCache rpc_socket_type must be 'rdma' or 'tcp'")

        template = extra.get("config_file", os.environ.get("GCACHE_CONFIG_FILE"))
        if template is None and Path("/sgl-workspace/gcache.flags").is_file():
            template = "/sgl-workspace/gcache.flags"
        if template is not None:
            template = _text("config_file", template)
            if not template:
                raise ValueError("GCache config_file must not be empty")

        # The SDK creates an admin Unix socket under log_dir. Keep the default
        # short enough for sockaddr_un, independently of the checkout location.
        default_dir = str(Path(tempfile.gettempdir()) / f"gcache-{os.getuid()}")
        paths = {}
        for name in ("config_dir", "log_dir"):
            value = _text(name, extra.get(name, default_dir))
            if not value:
                raise ValueError(f"GCache {name} must not be empty")
            paths[name] = Path(value)

        return cls(
            master_addr=master_addr,
            cache_group=cache_group,
            client_ifname=client_ifname,
            client_threads=_integer(
                "GCACHE_CLIENT_THREADS", os.environ.get("GCACHE_CLIENT_THREADS", "8"), 1
            ),
            traffic_class=_integer(
                "GCACHE_TRAFFIC_CLASS",
                os.environ.get("GCACHE_TRAFFIC_CLASS", "-1"),
                -1,
                255,
            ),
            rdma_allowed_devices=_text("HPN_IB_HCA", os.environ.get("HPN_IB_HCA", "")),
            rpc_socket_type=transport,
            timeout_us=_integer("timeout_us", extra.get("timeout_us", 0), 0),
            config_file=Path(template) if template is not None else None,
            **paths,
        )

    def render(self) -> str:
        template = (
            self.config_file.read_text(encoding="utf-8")
            if self.config_file
            else _DEFAULT_FLAGS
        )
        overrides = {
            "master_addr": self.master_addr,
            "cache_group": self.cache_group,
            "client_ifname": self.client_ifname,
            "client_threads": self.client_threads,
            "traffic_class": self.traffic_class,
            "rpc_socket_type": self.rpc_socket_type,
            "log_dir": str(self.log_dir),
        }
        if self.rdma_allowed_devices:
            overrides["rdma_allowed_devices"] = self.rdma_allowed_devices
        return render_flags(template, overrides)

    def write_flags(
        self, dp_rank: int, tp_rank: int, *, contents: str | None = None
    ) -> Path:
        contents = self.render() if contents is None else contents
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.config_dir,
            prefix=f"gcache.dp{dp_rank}tp{tp_rank}.",
            suffix=".flags",
            delete=False,
        ) as flags:
            flags.write(contents)
            return Path(flags.name)
