import os
from dataclasses import dataclass, field
from transformers import AutoConfig


MODEL_PATHS = {
    "qwen3-14B": "/home/zhj/huggingface/Qwen3-14B",
    "qwen3-0.6B": "/home/zhj/huggingface/Qwen3-0.6B",
}


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    name: str = "target"
    gpu_ids: tuple[int, ...] = (0,)
    worker_timeout: float = 600.0
    graph_memory_reserve_mb: int = 256
    target_verify_graph: bool = True
    # Filled from EngineConfig before workers are spawned.
    speculative_k: int = 0

    def __post_init__(self):
        if self.graph_memory_reserve_mb < 0:
            raise ValueError("graph_memory_reserve_mb must be nonnegative")
        self.model = os.path.abspath(os.path.expanduser(self.model))
        self.gpu_ids = tuple(self.gpu_ids)
        if not os.path.isdir(self.model):
            raise ValueError(f"{self.name}: local model directory does not exist: {self.model}")
        if self.name not in ("draft", "target"):
            raise ValueError("Scheduler name must be 'draft' or 'target'")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(f"{self.name}: gpu_ids must contain one GPU per TP rank")
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(i < 0 for i in self.gpu_ids):
            raise ValueError(f"{self.name}: GPU IDs must be unique and nonnegative")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.max_model_len) <= 0:
            raise ValueError("Batch and sequence limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1 or self.worker_timeout <= 0:
            raise ValueError("Invalid GPU memory utilization or worker timeout")
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


@dataclass(slots=True)
class EngineConfig:
    speculative_k: int = 5
    batch: int = 8
    draft: Config | None = None
    target: Config = field(default_factory=lambda: Config(
        model=MODEL_PATHS["qwen3-14B"], name="target", gpu_ids=(0, 1), tensor_parallel_size=2,
        enforce_eager=False,
    ))

    def __post_init__(self):
        if self.speculative_k < 0 or self.batch < 1:
            raise ValueError("K must be nonnegative and batch must be positive")
        if self.target.name != "target":
            raise ValueError("Target must use name='target'")
        self.target.speculative_k = self.speculative_k
        if self.speculative_k:
            # enforce_eager controls native K=0; target_verify_graph controls verify.
            self.target.enforce_eager = True
            if self.draft is None:
                self.draft = Config(model=MODEL_PATHS["qwen3-0.6B"], name="draft",
                                    gpu_ids=(2,), tensor_parallel_size=1, enforce_eager=False)
            if self.draft.name != "draft":
                raise ValueError("Draft must use name='draft'")
            if set(self.draft.gpu_ids) & set(self.target.gpu_ids):
                raise ValueError("Draft and target must use separate GPUs")
        elif self.target.tensor_parallel_size != 2:
            raise ValueError("K=0 uses the ordinary target TP=2 decode path")
        for config in ([self.target, self.draft] if self.speculative_k else [self.target]):
            config.max_num_seqs = min(config.max_num_seqs, self.batch)
