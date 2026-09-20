# GPTQ Marlin Inference Profiling

GPU profiling and bottleneck analysis of **GPTQ INT4 LLM inference** with **vLLM + Marlin**, using NVIDIA **Nsight Systems** and **Nsight Compute**.

This project investigates where time is spent during autoregressive decode and why the dominant quantized GEMM kernels are not fully utilizing the GPU.

The focus is not simply benchmarking INT4 versus FP16. The workflow is:

```text
Quantized inference
        ↓
System-level profiling
        ↓
Identify dominant kernels
        ↓
Kernel-level profiling
        ↓
Classify the bottleneck
        ↓
Test targeted optimization ideas
        ↓
Re-profile and validate
```

## Key Result

For steady-state GPTQ INT4 decode on an NVIDIA Tesla T4, a representative Marlin kernel showed:

| Metric                                 |   Result |
| -------------------------------------- | -------: |
| Compute (SM) throughput                |   28.09% |
| Memory throughput                      |   14.49% |
| Registers / thread                     |      198 |
| Dynamic shared memory / block          | 65.54 KB |
| Theoretical occupancy                  |      25% |
| Achieved occupancy                     |   26.06% |
| Eligible warps / scheduler             |     0.26 |
| Scheduler cycles with no eligible warp |   82.01% |

The kernel was therefore **not saturating either compute throughput or DRAM bandwidth**.

The evidence instead points to a **latency-hiding / low-residency bottleneck**:

```text
high register usage
+
high shared-memory usage
        ↓
limited block residency
        ↓
~25% occupancy
        ↓
few eligible warps
        ↓
poor latency hiding
        ↓
underutilized compute and memory resources
```

## Experimental Setup

* **Model:** `JunHowie/Qwen3-0.6B-GPTQ-Int4`
* **Runtime:** vLLM 0.29.0
* **GPU:** NVIDIA Tesla T4, compute capability 7.5
* **Activation dtype:** FP16
* **Quantization:** GPTQ INT4
* **Quantized linear backend:** Marlin
* **Attention backend:** Triton
* **Profilers:** Nsight Systems + Nsight Compute

The main profiling workload used:

```text
Input length:   32 tokens
Output length: 256 tokens
Batch size:     1
```

The short prompt and longer generation were chosen to emphasize autoregressive decode.

## 1. System-Level Profiling

Nsight Systems was first used to identify the dominant GPU kernel families.

The initial profile included startup, compilation, CUDA Graph capture, warmup, and inference. That trace was useful for discovering kernels, but not for steady-state performance analysis.

The profiling procedure was therefore corrected to isolate a request after model initialization and warmup.

In the steady-state trace, the major GPU costs were approximately:

| Kernel family              | GPU kernel time |
| -------------------------- | --------------: |
| `gemvx`                    |          ~28.5% |
| Marlin variant #1          |          ~21.9% |
| Marlin variant #2          |          ~17.3% |
| `kernel_unified_attention` |          ~14.7% |

The two dominant Marlin variants together represented roughly **39% of GPU kernel time**, making the compressed linear path a primary analysis target.

## 2. Kernel-Level Analysis

Nsight Compute was then used to determine **why** the Marlin kernel was expensive.

The first NCU capture selected a warmup/startup Marlin invocation and was not representative of normal decode.

A later Marlin invocation was therefore captured after skipping thousands of earlier matching launches.

That steady-state kernel showed:

* low compute utilization,
* low memory-bandwidth utilization,
* high register usage,
* high shared-memory usage,
* low occupancy,
* and very few eligible warps.

The most important scheduler result was:

```text
No Eligible = 82.01%
```

In other words, for most scheduler cycles there was no warp ready to issue its next instruction.

This is consistent with insufficient latency hiding rather than simple compute or DRAM saturation.

## 3. Exploratory Optimization

One experimental Marlin configuration was tested:

```bash
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

A controlled serving benchmark used the same:

* model,
* input/output lengths,
* concurrency,
* number of requests,
* and warmup procedure.

Initial results were:

| Metric            |     Baseline |   Atomic Add |
| ----------------- | -----------: | -----------: |
| Mean TPOT         |      4.61 ms |      4.88 ms |
| Output throughput | 212.19 tok/s | 201.06 tok/s |
| Mean ITL          |      4.61 ms |      4.88 ms |

Atomic-add therefore did **not** produce a defensible end-to-end decode improvement for this workload.

Kernel traces also showed run-to-run changes in unrelated kernels such as attention, making it unsafe to attribute lower individual Marlin timings solely to the atomic-add configuration.

This optimization path was therefore abandoned.

## Current Status

### Completed

* GPTQ INT4 inference through vLLM/Marlin
* steady-state decode isolation
* system-level GPU profiling
* dominant kernel identification
* Marlin hardware-counter analysis
* occupancy and scheduler diagnosis
* one profile-motivated optimization experiment
* end-to-end validation of that experiment

### Open

The next optimization stage remains intentionally open.

Potential directions include:

* investigating Marlin tile/resource configurations,
* testing another quantized GEMM execution path,
* reducing register or shared-memory pressure,
* implementing a small W4A16 CUDA/Triton kernel for a representative model shape,
* or investigating the expensive `gemvx` path.

The next optimization should be selected from profiler evidence rather than from benchmark results alone.

## Repository Contents

```text
.
├── README.md
├── GPTQ_Marlin_bottlenech.ipynb
├── figures/
│   ├── nsys-kernel-summary.png
│   ├── ncu-scheduler-occupancy.png
│   └── ncu-launch-resource-usage.png
└── results/
    ├── nsys-kernel-summary.txt
    └── ncu-marlin-summary.txt
```

## Main Takeaway

The main finding is not a final speedup percentage.

It is a concrete characterization of a compressed-inference bottleneck:

> On this T4 decode workload, the profiled Marlin kernel is not primarily limited by peak compute or DRAM bandwidth. Its register and shared-memory footprint restricts residency, resulting in low occupancy and very few eligible warps, which limits the GPU's ability to hide latency.

The next phase is to determine whether modifying the quantized execution path can improve that behavior and produce a measurable reduction in end-to-end decode latency.

I would use this as the main `README.md`; the longer blog post can go under `docs/` as `profiling-analysis.md`.
