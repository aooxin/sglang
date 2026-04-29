import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_blackwell_system,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, suite="nightly-8-gpu-b200", nightly=True)

MIMO_V2_PRO_MODEL_PATH = "XiaomiMiMo/MiMo-V2.5-Pro"

BASE_ARGS = [
    "--pp-size",
    "1",
    "--tp-size",
    "8",
    "--page-size",
    "128",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.75",
    "--max-running-requests",
    "128",
    "--chunked-prefill-size",
    "16384",
    "--reasoning-parser",
    "mimo",
    "--tool-call-parser",
    "mimo",
    "--context-length",
    "1048576",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 64}',
    "--attention-backend",
    "fa4",
]

EAGLE_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
]


class _MiMoV2ProB200Base(CustomTestCase):
    model = MIMO_V2_PRO_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args = BASE_ARGS
    env = None

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MIMO_V2_PRO_MODEL_PATH)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


@unittest.skipIf(not is_blackwell_system(), "Requires B200")
class TestMiMoV2ProB200(GSM8KMixin, _MiMoV2ProB200Base):
    gsm8k_accuracy_thres = 0.75
    gsm8k_num_questions = 200
    gsm8k_num_threads = 200


@unittest.skipIf(not is_blackwell_system(), "Requires B200")
class TestMiMoV2ProB200MTP(GSM8KMixin, _MiMoV2ProB200Base):
    gsm8k_accuracy_thres = 0.75
    gsm8k_num_questions = 200
    gsm8k_num_threads = 200
    other_args = BASE_ARGS + EAGLE_ARGS
    env = {"SGLANG_ENABLE_SPEC_V2": "1"}


if __name__ == "__main__":
    unittest.main()
