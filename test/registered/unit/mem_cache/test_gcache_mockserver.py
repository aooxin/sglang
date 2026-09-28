"""The KV test service really owns bytes in a separate, reachable process."""

import ctypes
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_mock_server import GCacheMockServer, NetworkGCacheClient

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_independent_clients_exchange_bytes_through_server(tmp_path):
    with GCacheMockServer() as server:
        assert server.pid != os.getpid()
        flags = tmp_path / "test.flags"
        flags.write_text(f"--master_addr={server.address}\n--cache_group=test\n")
        writer = NetworkGCacheClient(conf_file=flags)
        reader = NetworkGCacheClient(conf_file=flags)
        payload = b"\x00\xffreal network KV\x00"
        source = ctypes.create_string_buffer(payload, len(payload))
        dest = ctypes.create_string_buffer(len(payload))
        writer.register_memory(ctypes.addressof(source), len(payload))
        reader.register_memory(ctypes.addressof(dest), len(payload))
        assert writer.batch_set(["k"], [ctypes.addressof(source)], [len(payload)]) == [
            True
        ]
        # Prove the server copied and owns the value, independent of the writer.
        ctypes.memset(ctypes.addressof(source), 0, len(payload))
        writer.close()
        assert reader.batch_exists(["k"], size_list=[len(payload)]) == [True]
        assert reader.batch_get(["k"], [ctypes.addressof(dest)], [len(payload)]) == [
            True
        ]
        assert dest.raw == payload
        assert server.objects == {"k": payload}
        assert server.snapshot()["pid"] == server.pid
        assert [call["operation"] for call in server.calls] == ["set", "exists", "get"]
        server.close()
        with pytest.raises(OSError):
            reader.batch_exists(["k"], size_list=[len(payload)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
