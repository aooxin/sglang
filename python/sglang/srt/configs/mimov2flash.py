from sglang.srt.sampling.custom_logit_processor import (
    MiMoV2FlashNoRepeatNGramLogitProcessor,
)

DEFAULT_CUSTOM_LOGIT_PROCESSOR = MiMoV2FlashNoRepeatNGramLogitProcessor.to_str()
