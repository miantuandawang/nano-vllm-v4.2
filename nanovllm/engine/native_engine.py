import atexit
import os

from nanovllm.native.engine.llm_engine import LLMEngine as UpstreamEngine
from nanovllm.native.sampling_params import SamplingParams


class NativeTP2Engine(UpstreamEngine):
    """Configuration/lifecycle adapter; computation and scheduling are upstream."""
    sampling_params = SamplingParams

    def __init__(self, config):
        target = config.target
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target.gpu_ids))
        super().__init__(
            target.model, tensor_parallel_size=2,
            enforce_eager=target.enforce_eager,
            max_num_seqs=min(config.batch, target.max_num_seqs),
            max_num_batched_tokens=target.max_num_batched_tokens,
            max_model_len=target.max_model_len,
            gpu_memory_utilization=target.gpu_memory_utilization,
            kvcache_block_size=target.kvcache_block_size,
        )

    def add_request(self, prompt, sampling_params):
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("K=0 requires NativeSamplingParams (native greedy/sampling parameters)")
        return super().add_request(prompt, sampling_params)

    def exit(self):
        if hasattr(self, "model_runner"):
            super().exit()
        atexit.unregister(self.exit)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
