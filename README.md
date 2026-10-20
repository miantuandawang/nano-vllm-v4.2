# Nano-vLLM V4.2: greedy speculative decoding

An experimental fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), retaining its MIT license and attribution.
Both K=0 and K>0 use greedy argmax sampling. Default placement: Qwen3-14B on GPUs 0/1
(TP=2); Qwen3-0.6B on GPU 2 (TP=1). Local paths are in `nanovllm/config.py`.

```bash
python example.py --k 5 --batch 8
python example.py --k 0 --batch 8
```

`EngineConfig(speculative_k=5, batch=8)` configures the same settings in Python.
The batch limit applies per scheduler dispatch, including mixed prefill/work
batches. Smaller max_num_seqs and the token budget may further limit selection.
K=0 starts only target TP=2 and does not require draft files or a draft worker.


## 安装与运行（V4.2）

V4.2 在 V4.1 上仅增加 Target eager prefill 的 LM Head 行选择：Transformer
仍处理完整未缓存输入，LM Head 只处理 prefill 的最后一行；verify 保留所需预测行。
详见 [V4.2_MODIFICATIONS.md](V4.2_MODIFICATIONS.md)。

已运行环境：Linux、Python 3.12.12、PyTorch 2.8.0+cu128、Triton 3.4.0、
FlashAttention 2.8.3.post1、Transformers 4.57.6，RTX 4090。
`requirements.txt` 固定已记录的核心依赖版本；未记录精确版本的 xxhash/tqdm
使用版本范围，因此它不是完整的传递依赖锁文件。FlashAttention 源码构建需要
匹配的 CUDA toolkit / nvcc 和编译工具；本项目不包含模型权重。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
# 先安装 Torch，供 FlashAttention 构建使用。
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps
```

使用已有本地模型目录，显式覆盖源码中的原工作站默认路径：

```bash
# K>0：Target TP=2 (GPU 0/1)，Draft TP=1 (GPU 2)
python example.py --target-model-path /models/Qwen3-14B \
  --draft-model-path /models/Qwen3-0.6B \
  --target-gpus 0 1 --draft-gpus 2 --batch 16 --k 4 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --target-gpu-memory-utilization 0.98 --draft-gpu-memory-utilization 0.98

# K=0：仅启动原生 TP=2 / CUDA Graph 路径，采样改为 greedy。
python example.py --target-model-path /models/Qwen3-14B \
  --target-gpus 0 1 --batch 16 --k 0 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --target-gpu-memory-utilization 0.98
```

0.98 是实验设置，不是通用的容量保证。源码默认值未改动；在新硬件或新负载上
按可用显存设置参数。K>0 prefill 保持 eager，Draft 单 token 与完整 K 候选的
Target verify 使用已有 CUDA Graph；不满足 Graph 形状的请求走既有 eager 路径。

`bench.py` 是随机 token 的简易吞吐基准，不是 Spec-Bench/HumanEval。
Spec-Bench/HumanEval 的 72 次实验已在服务器执行，完整输入输出、日志及性能数据
另行归档，没有放入源码仓库。实验完成不等于与 V4.1 逐 token 一致；LM Head
矩阵形状变化可能导致浮点舍入及 greedy 结果差异。

本次 GitHub 打包不修改推理代码，也不重新执行 GPU 测试。

## Queues and lifecycle

States belong to each Sequence. Queues contain request IDs, and two model copies
of a request share an ID while retaining independent KV blocks.

Target uses prefill, running, verify and pausing. Draft uses prefill, running,
drafting and pausing. The unused role-specific queue stays empty in the shared
scheduler implementation.

1. An incoming request is created only in target.prefill. Draft has no copy yet.
2. Target selects verify first, then fills spare dispatch slots from prefill.
   A new request moves prefill -> running. Prompt prefill yields its first greedy
   token directly from the last prompt logits; no redundant decode is required.
3. If this token ends the request, target completes locally. Otherwise target
   moves running -> pausing and calls draft.receive_prefill with request_id,
   prompt tokens, first token and output parameters.
4. Draft now creates its Sequence in prefill. Its initial queue was empty.
   Draft selects drafting first and then fills spare slots with prefill.
5. A draft prefill task processes prompt + target's first token, writes their KV,
   and uses those logits to produce candidate 1. The same task continues until K
   candidates, EOS (unless ignore_eos), or the remaining output allowance.
   There is no intermediate prefill -> drafting round trip.
6. Draft moves running -> pausing and directly calls target.receive_draft.
   Target moves pausing -> verify. Target accepts the matching greedy prefix and
   adds a replacement at the first mismatch or a bonus when all candidates match.
7. Target commits results and trims its cache, then sends accepted_count and
   next_token directly to draft.receive_target_result. Draft updates its confirmed
   context, trims its own cache, and moves pausing -> drafting for the next round.
8. Target-confirmed EOS/output limit directly releases both copies and KV blocks.
   A draft EOS is only a candidate until target accepts it.

```text
target: prefill -> running -> pausing -> verify -> running -> pausing ...
draft:  (created after first target token)
        prefill -> running [prefill + K candidates] -> pausing
        pausing -> drafting -> running [K candidates] -> pausing ...
```

High-priority tasks consume a shared batch/token budget first. A deferred
high-priority task prevents admission of additional prefill into that dispatch.
Mixed draft tasks retain the batch cap throughout their internal loop; sequences
that hit EOS/limit leave the loop. Graph-eligible target verification runs first; remaining verification/prefill uses eager in the same capped dispatch.
Engine polls completions and drives scheduling; it never relays peer payloads.

## Allocation and rollback

There is no max-output-length reservation or full-lifetime admission accounting.
Only missing blocks for the current task are allocated:

- Target prefill: prompt length.
- Target verification: confirmed length + received candidate count.
- Draft round, including initial prefill: confirmed length + round limit - 1.
- Ordinary target decode: current confirmed length.

The last generated token has not yet been fed through the model. After accepting
A candidates from confirmed length L, target retains cache through L+A, and
draft retains min(actual cached length, L+A). The replacement/bonus is confirmed
but remains uncached. Next round computes that missing suffix. Unused complete
tail blocks are returned after early draft EOS and verification rollback.

Blocks are exclusive to requests; prefix sharing remains disabled. Draft single-token decode uses CUDA Graphs by default; full-width target verification also uses CUDA Graphs by default, with eager fallback for other shapes.
Requests waiting for a peer retain cache. There is no eviction/recomputation;
if all work is blocked by KV capacity and no worker can make progress, the engine
reports capacity exhaustion instead of spinning. No full-length reservation is
used to avoid this condition.

## Native K=0 TP=2 path

K=0 constructs `NativeTP2Engine`, a thin configuration/context-manager adapter
over `nanovllm.native.engine.llm_engine.LLMEngine`. It does not construct either
custom scheduler, the custom ModelRunnerGroup, or the speculative model runner.

`nanovllm/native/` is copied from GeeeekExplorer/nano-vllm commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`; only absolute package imports are
relocated. Native scheduler, block manager, prefix cache, chunked prefill,
preemption, kernels, NCCL initialization and shared-memory TP dispatch
are preserved. Only the native sampler and sampling defaults are adapted to greedy. Native rank 0 executes in the host process and rank 1 is spawned.
GPU visibility is set to the configured target GPU IDs before initialization;
start this mode in a fresh process before creating any CUDA context.

`batch` maps to upstream `max_num_seqs`. Native CUDA Graphs are enabled by default
for the CLI/default EngineConfig; `--enforce-eager` explicitly disables them.
The upstream implementation's constraints (including graph capture batch sizes)
are retained, not silently patched. There are no five-queue policies in K=0.

```bash
python example.py --k 0 --batch 8
python example.py --k 0 --batch 8 --enforce-eager
```

The pinned upstream sampler originally rejected temperature=0. Per the updated
requirement, native SamplingParams now defaults to zero and the native compiled
sampler returns argmax directly. Both modes support only greedy sampling.
This is the only computational change to the pinned native path; its scheduler,
TP, cache and CUDA Graph execution remain upstream.

```python
from nanovllm import LLM, EngineConfig, NativeSamplingParams

if __name__ == "__main__":
    with LLM(EngineConfig(speculative_k=0, batch=8)) as llm:
        outputs = llm.generate(["Hello"], NativeSamplingParams(max_tokens=128))
```

For mode-independent callers, `llm.sampling_params(...)` selects the appropriate
parameter class. Example and benchmark use this entry point. K=0 retains native
low-level API behavior: step() returns `(outputs, num_tokens)`, and add_request()
does not return an ID. K>0 keeps its existing speculative API.

Model paths remain local; no weights were downloaded. No tests, model execution
or benchmarks were run for this change.


## 4.0 draft CUDA Graph

See [CHANGES_4.0.md](CHANGES_4.0.md) for the implementation notes in Chinese.
K>0 uses eager draft prefill/catch-up and graph replay for single-token steps.
Only the model backbone is captured; LM head and greedy argmax remain outside.
Buckets follow upstream 1,2,4,8,16,32,48,...,512, clipped to the configured batch
limit with a final boundary bucket when needed. Replay rounds up the active batch.

```bash
python example.py --k 5 --batch 24
python example.py --k 5 --batch 24 --draft-enforce-eager
python example.py --k 5 --batch 24 --draft-graph-memory-reserve-mb 512
```

Graph memory headroom defaults to 256 MiB per draft rank and is deducted from
KV capacity before allocation. This is a configurable estimate, not a guarantee
of capture capacity; no automatic retries or silent fallback on capture failure.
The K=0 native package is copied unchanged from 2.0.


## 4.1 target verify CUDA Graph

See [CHANGES_4.1.md](CHANGES_4.1.md) for the detailed Chinese change report.
K>0 target captures only the Transformer backbone using the existing paged varlen
attention. LM head, greedy and acceptance remain outside capture. Native-style
request buckets (1,2,4,8,16,32,48,...) round up the actual verify batch; each row
contains K+1 queries. Buckets are capped by batch, 512 and the configured token
budget, with a final boundary bucket as needed.

Only complete K-candidate requests with one uncached confirmed token use replay.
Short drafts, catch-up and prefill remain eager. Padding reads dedicated zero KV
scratch blocks, skips cache writes via slot=-1, and never enters sampling. Scratch
blocks are excluded from the scheduler's free pool. No per-request maximum-length
cache reservation is introduced.

```bash
python example.py --k 6 --batch 16
python example.py --k 6 --batch 16 --target-enforce-eager
python example.py --k 6 --batch 16 --target-graph-memory-reserve-mb 512
```

Target headroom defaults to 256 MiB per rank and is a configurable estimate, not
a measured memory requirement. Capture errors are reported through the existing
worker channel. `--enforce-eager` retains its K=0 meaning; draft retains its own
switch. `Config.target_verify_graph` controls K>0 target graph dispatch.
No tests, compilation checks, model execution or benchmarks were run for 4.1.
