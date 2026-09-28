"""GCache configuration must preserve NIC selection and emit SDK overrides."""

import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def configured_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("GCACHE_") or name in ("HPN_IB_IFNAME", "HPN_IB_HCA"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GCACHE_MASTER_ADDR", " master.test ")
    monkeypatch.setenv("GCACHE_CACHE_GROUP", " group-test ")
    monkeypatch.setenv("HPN_IB_IFNAME", "eth1,eth2")
    return monkeypatch


def test_config_defaults_and_whitespace(configured_env):
    from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

    config = GCacheClientConfig.from_env()
    assert config.master_addr == "master.test"
    assert config.cache_group == "group-test"
    assert config.client_ifname == "eth1,eth2"
    assert config.client_threads == 8
    assert config.traffic_class == -1
    assert config.rpc_socket_type == "rdma"


@pytest.mark.parametrize(
    "hpn,legacy,want",
    [
        ("eth1,eth2", "eth3,eth4", "eth1,eth2"),
        ("eth1,eth2", "eth3", "eth3"),
        (None, "eth3,eth4", "eth3,eth4"),
    ],
)
def test_legacy_nic_precedence(configured_env, hpn, legacy, want):
    from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

    if hpn is None:
        configured_env.delenv("HPN_IB_IFNAME")
    else:
        configured_env.setenv("HPN_IB_IFNAME", hpn)
    configured_env.setenv("GCACHE_CLIENT_IFNAME", legacy)
    assert GCacheClientConfig.from_env().client_ifname == want


def test_renderer_updates_and_appends_flags():
    from sglang.srt.mem_cache.storage.gcache.config import render_flags

    template = "# keep this comment\n--client_threads=2\n--verify_checksum=false\n"
    rendered = render_flags(
        template, {"client_threads": 8, "rdma_allowed_devices": "mlx5_0,mlx5_1"}
    )
    assert rendered == (
        "# keep this comment\n--client_threads=8\n--verify_checksum=false\n"
        "--rdma_allowed_devices=mlx5_0,mlx5_1\n"
    )


def test_flags_file_contains_hca_rank_and_local_paths(configured_env, tmp_path):
    from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

    configured_env.setenv("HPN_IB_HCA", "mlx5_1")
    template = tmp_path / "template.flags"
    template.write_text("--client_threads=1\n--verify_checksum=true\n")
    config = GCacheClientConfig.from_env(
        {
            "config_file": str(template),
            "config_dir": str(tmp_path / "flags"),
            "log_dir": str(tmp_path / "logs"),
            "rpc_socket_type": "tcp",
        }
    )
    path = config.write_flags(dp_rank=2, tp_rank=3)
    flags = dict(line[2:].split("=", 1) for line in path.read_text().splitlines())
    assert flags["rdma_allowed_devices"] == "mlx5_1"
    assert flags["master_addr"] == "master.test"
    assert flags["cache_group"] == "group-test"
    assert flags["client_threads"] == "8"
    assert flags["verify_checksum"] == "true"
    assert flags["rpc_socket_type"] == "tcp"
    assert flags["log_dir"] == str(tmp_path / "logs")
    assert "dp2tp3" in path.name
    assert path != config.write_flags(dp_rank=2, tp_rank=3)


@pytest.mark.parametrize(
    "name,value",
    [
        ("GCACHE_MASTER_ADDR", ""),
        ("GCACHE_CACHE_GROUP", "  "),
        ("HPN_IB_IFNAME", ""),
        ("GCACHE_CLIENT_THREADS", "0"),
        ("GCACHE_CLIENT_THREADS", "not-a-number"),
        ("GCACHE_TRAFFIC_CLASS", "256"),
        ("GCACHE_MASTER_ADDR", "host\n--cache_group=other"),
    ],
)
def test_invalid_config_fails_before_sdk_initialization(configured_env, name, value):
    from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

    configured_env.setenv(name, value)
    with pytest.raises(ValueError):
        GCacheClientConfig.from_env()


def test_extra_config_does_not_mutate_environment(configured_env):
    from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

    before = dict(os.environ)
    config = GCacheClientConfig.from_env({"rpc_socket_type": "tcp", "timeout_us": 42})
    assert config.rpc_socket_type == "tcp"
    assert config.timeout_us == 42
    assert dict(os.environ) == before
    with pytest.raises(ValueError, match="unknown|Unknown|unsupported|Unsupported"):
        GCacheClientConfig.from_env({"typo_config": True})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
