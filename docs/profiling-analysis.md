Use the filename **`docs/profiling-analysis.md`**. This version is intentionally more technical than the README and documents both the successful analysis and the measurement corrections.

# Profiling GPTQ INT4 LLM Inference: Diagnosing a Marlin Decode Bottleneck on NVIDIA T4

## Abstract

This document describes a GPU profiling investigation of GPTQ INT4 language-model inference using **Qwen3-0.6B-GPTQ-Int4**, **vLLM 0.29.0**, and the **Marlin** quantized matrix-multiplication backend on an NVIDIA Tesla T4.

The objective was not simply to compare tokens-per-second numbers between quantized and unquantized models. Instead, the work focused on understanding the actual execution path of compressed LLM inference:

1. identify which CUDA kernels dominate autoregressive decode,
2. isolate steady-state inference from initialization and compilation,
3. inspect the dominant quantized GEMM kernel using hardware counters,
4. classify the bottleneck as compute-, memory-, launch-, or latency-related,
5. test an optimization hypothesis derived from the runtime and profiling results,
6. validate any change at both the kernel and end-to-end serving levels.

The main result is a hardware-level diagnosis of a representative steady-state Marlin decode kernel.

The profiled kernel achieved only approximately:

* **28.09% Compute (SM) throughput**
* **14.49% memory throughput**
* **26.06% achieved occupancy**

while using:

* **198 registers per thread**
* **65.54 KB dynamic shared memory per block**

The scheduler reported **82.01% of cycles with no eligible warp** and only **0.26 eligible warps per scheduler** on average. 

These measurements indicate that the kernel is neither saturating peak arithmetic throughput nor DRAM bandwidth. Instead, the observed behavior is consistent with **low residency and insufficient latency hiding**, caused in part by register and shared-memory pressure.

A subsequent experimental Marlin configuration using `VLLM_MARLIN_USE_ATOMIC_ADD=1` did not produce a defensible end-to-end improvement and was therefore abandoned rather than presented as a successful optimization.

The optimization phase remains open.

---

# 1. Motivation

Weight-only quantization is often described primarily as a memory optimization.

Reducing model weights from FP16/BF16 to INT4 can substantially reduce:

* model storage,
* GPU memory consumption,
* weight-transfer traffic,
* and potentially inference latency.

However, lower precision does not guarantee proportional speedup.

A compressed linear layer introduces its own execution costs:

* packed-weight loading,
* unpacking and dequantization,
* scale application,
* quantized GEMM implementation details,
* reduction behavior,
* shared-memory staging,
* register pressure,
* tile selection,
* and synchronization.

As a result, the useful question is not simply:

> Is GPTQ faster than FP16?

A more useful performance-engineering question is:

> **What does the GPU actually execute during GPTQ decode, and what prevents the expensive kernels from running faster?**

This project therefore uses a profiler-first workflow:

```text
GPTQ INT4 inference
        ↓
Nsight Systems
        ↓
Identify dominant kernel families
        ↓
Isolate steady-state decode
        ↓
Select an important quantized kernel
        ↓
Nsight Compute
        ↓
Measure hardware utilization
        ↓
Classify the bottleneck
        ↓
Form an optimization hypothesis
        ↓
Validate at kernel + application level
```

The distinction between **profiling** and **benchmarking** is important throughout this work.

A benchmark answers:

> How fast was this configuration?

A profiler should help answer:

> Why was it this fast?

---

# 2. Experimental Environment

## 2.1 Model

The model used throughout the experiment is:

```text
JunHowie/Qwen3-0.6B-GPTQ-Int4
```

The checkpoint is a GPTQ INT4 quantized Qwen3-0.6B model.

A relatively small model was chosen deliberately. The objective of this phase was rapid iteration on the inference execution path rather than model-quality evaluation or maximum-throughput deployment.

A small checkpoint makes it practical to repeatedly:

* start new vLLM processes,
* collect profiler traces,
* change execution configurations,
* and run Nsight Compute replay passes.

---

## 2.2 Inference Runtime

The model was executed with:

```text
vLLM 0.29.0
```

At initialization, vLLM reported:

```text
Using MarlinLinearKernel for AutoGPTQLinearMethod
```

confirming that the GPTQ linear layers were being dispatched through the Marlin quantized linear backend. 

This confirmation is important.

Without it, an INT4 checkpoint alone would not prove that the experiment is actually exercising the intended optimized quantized kernel path.

---

## 2.3 GPU

The profiling environment used:

```text
NVIDIA Tesla T4
Compute Capability: 7.5
```

Because the Tesla T4 does not support BF16 execution, vLLM automatically fell back to FP16 activations.

The runtime also reported that FlashAttention 2 was unavailable on the device and selected the Triton attention backend instead. 

Therefore the main execution configuration can be summarized as:

```text
GPTQ INT4 weights
        +
FP16 activations
        ↓
Marlin quantized linear kernels

Attention
        ↓
TRITON_ATTN
```

These hardware-specific choices matter. The conclusions in this document describe this particular T4 execution regime and should not automatically be generalized to Ampere, Ada, Hopper, or Blackwell GPUs.

---

# 3. Primary Profiling Workload

The main workload was intentionally decode-heavy:

```text
Input length:   32 tokens
Output length: 256 tokens
Batch size:     1
```

The short input reduces the relative importance of long-prompt prefill.

The longer output creates many sequential autoregressive decode iterations.

This makes the workload useful for investigating low-batch decode behavior, where each new token repeatedly executes:

* transformer linear layers,
* attention,
* normalization and activation operations,
* KV-cache updates,
* sampling,
* and the final output projection.

The project is therefore not attempting to characterize every possible serving regime.

In particular, it does not yet attempt to characterize:

* high-concurrency serving,
* large batch sizes,
* long-context prefill,
* speculative decoding,
* tensor parallelism,
* or multi-GPU inference.

The current focus is specifically **single-request autoregressive decode**.

---

# 4. Why Two Profilers Were Needed

Two NVIDIA profiling tools were used because they answer different classes of questions.

## 4.1 Nsight Systems

Nsight Systems is used for system-level timing and execution structure.

It answers questions such as:

* Which kernels are running?
* How frequently are they launched?
* How much aggregate GPU time does each kernel family consume?
* Are there CPU/GPU gaps?
* Are CUDA Graphs involved?
* Is initialization contaminating the measurement?
* Which kernel should be investigated next?

A useful mental model is:

> **Nsight Systems tells us where the time goes.**

---

## 4.2 Nsight Compute

Nsight Compute operates at the individual CUDA-kernel level.

It can inspect:

* compute throughput,
* DRAM throughput,
* cache behavior,
* register allocation,
* shared-memory allocation,
* occupancy,
* active and eligible warps,
* scheduler issue behavior,
* warp stalls,
* instruction behavior,
* and memory-access efficiency.

A corresponding mental model is:

> **Nsight Compute helps explain why a selected kernel behaves as it does.**

The profiling workflow therefore became:

```text
Nsight Systems
      ↓
Find important kernel

Nsight Compute
      ↓
Explain important kernel
```

---

# 5. Initial Nsight Systems Profile

The first Nsight Systems experiment profiled the entire command running the decode benchmark.

This produced a valid profiler trace, but the measurement region was too broad.

It contained more than steady-state generation.

Depending on the run, the trace could include:

* process startup,
* model loading,
* PyTorch/vLLM initialization,
* JIT compilation,
* `torch.compile`,
* CUDA Graph setup and capture,
* warmup execution,
* and finally the requested inference iteration.

This is a common profiling trap.

The profiler can measure the execution perfectly while the experiment measures the wrong part of the application.

The first trace was still useful for identifying candidate kernel families, but aggregate timing percentages from that trace could not safely be interpreted as steady-state decode behavior.

This led to the first major methodology correction.

---

# 6. Isolating Steady-State Decode

The experiment was redesigned so that startup work occurred outside the measured range.

The corrected sequence was:

```text
start vLLM server under Nsight Systems
        ↓
load model
        ↓
initialize runtime
        ↓
compile / warm up
        ↓
send unprofiled inference request
        ↓
START PROFILER RANGE
        ↓
send decode-heavy measured request
        ↓
STOP PROFILER RANGE
        ↓
terminate server
        ↓
analyze GPU kernel summary
```

Nsight Systems was configured with:

```text
--capture-range=cudaProfilerApi
```

and vLLM's CUDA profiler interface was used to define the request that should be recorded.

This produced a much cleaner representation of steady-state inference.

---

# 7. Steady-State Nsight Systems Results

The clean steady-state profile revealed four dominant kernel groups.

The top entries included approximately:

| Kernel                     | GPU Time | Instances |
| -------------------------- | -------: | --------: |
| `internal::gemvx::kernel`  |    28.3% |       256 |
| Marlin variant #1          |    21.9% |    14,280 |
| Marlin variant #2          |    17.4% |    14,280 |
| `kernel_unified_attention` |    14.8% |     7,168 |

Additional smaller contributors included:

* fused RMSNorm/Marlin kernels,
* `reduce_segments`,
* KV-cache operations,
* fused Triton activation kernels,
* and sampling-related operations. 

The first major conclusion was therefore:

> **The quantized Marlin linear path is a substantial component of steady-state decode GPU execution.**

The two dominant Marlin variants alone accounted for approximately:

```text
21.9% + 17.4% ≈ 39.3%
```

of GPU kernel time in this trace.

That made Marlin a natural target for hardware-counter analysis.

---

# 8. Kernel Launch Frequency

The launch counts were also useful.

The main attention kernel appeared:

```text
7,168 times
```

while the two dominant Marlin variants each appeared:

```text
14,280 times
```

during the decode request. 

These high counts are consistent with operations repeatedly executed across transformer layers and decode iterations.

This helped confirm that the trace was observing repeated model execution rather than being dominated by one-time runtime setup.

The key observation was not simply that individual Marlin kernels were slow.

It was that relatively small kernel costs were multiplied across **tens of thousands of invocations**.

For autoregressive inference, even tens of microseconds per repeatedly executed kernel can accumulate into a substantial fraction of per-token latency.

---

# 9. The GEMV Observation

The largest individual kernel family in the clean profile was:

```text
internal::gemvx::kernel
```

with:

```text
256 invocations
~369 ms aggregate GPU time
~28.3% of GPU kernel time
```



The exact model-level attribution of this kernel was not proven during this phase, so it is intentionally left unresolved.

That distinction is important.

The one-launch-per-output-step pattern makes some interpretations plausible, but a profiler investigation should separate:

* what the evidence directly demonstrates,
* from what is merely inferred.

For now, the correct conclusion is simply:

> `gemvx` is another major steady-state decode cost and remains an open investigation target.

This also illustrates a common consequence of optimizing compressed networks:

> Reducing one class of work can expose another previously secondary operation as the next system bottleneck.

---

# 10. Selecting Marlin for Deeper Analysis

Even though `gemvx` was the largest single kernel entry, the Marlin kernel family collectively consumed more GPU time and was directly tied to the compressed GPTQ execution path.

Since the project focuses on **compressed-model inference**, Marlin was selected as the primary kernel for deeper analysis.

The central question changed from:

> What takes time?

to:

> **Why does the Marlin kernel take this time?**

Possible explanations included:

1. DRAM bandwidth saturation,
2. Tensor Core / compute saturation,
3. insufficient parallelism,
4. occupancy limitations,
5. synchronization,
6. shared-memory behavior,
7. instruction dependencies,
8. register pressure,
9. or launch/runtime overhead.

Nsight Compute was required to distinguish among these possibilities.

---

# 11. Exploratory NVTX Attribution

Before the final Nsight Compute pass, layerwise NVTX tracing was briefly tested.

The goal was to determine whether the dominant CUDA kernels could be mapped cleanly back to individual model-level operations or transformer layers.

To enable detailed tracing, the experiment used an eager execution configuration.

This had an important consequence:

```text
--enforce-eager
```

disables the normal compiled/CUDA-Graph execution path.

Therefore this run was used only for attribution/debugging and was **not considered a valid performance measurement**.

The NVTX summary primarily exposed internal CUB ranges and did not provide useful Marlin-to-layer attribution.

Because the experiment changed the optimized runtime configuration without producing useful additional information, the NVTX path was deprioritized.

This avoided spending additional time on a diagnostic branch that was not helping answer the main bottleneck question.

---

# 12. First Nsight Compute Capture

The first successful Nsight Compute capture provided interesting data.

Among other things, it showed:

* high register allocation,
* large shared-memory usage,
* low occupancy,
* relatively high compute utilization,
* and few eligible warps.

However, the captured Marlin kernel took several milliseconds under that experiment.

That duration did not match the normal steady-state Marlin behavior visible in Nsight Systems, where the dominant Marlin kernels were on the order of tens of microseconds.

This suggested that NCU had captured a different Marlin invocation associated with startup or warmup.

The kernel itself was real.

The problem was again **which invocation had been selected**.

Therefore the first capture was treated as exploratory evidence rather than the final diagnosis.

This led to the second important measurement correction:

> Profile a later Marlin invocation that is more representative of steady-state decode.

---

# 13. Capturing a Representative Decode Marlin Kernel

To move deeper into the repeated inference execution, Nsight Compute was configured to skip a large number of matching `Marlin` launches and capture only one later invocation.

The experiment collected only the most relevant sections:

```text
SpeedOfLight
LaunchStats
Occupancy
SchedulerStats
WarpStateStats
```

A representative later kernel had:

```text
Grid:  (40, 1, 1)
Block: (256, 1, 1)
Duration: ~56.26 μs
```

which was much closer to the scale observed in the steady-state Nsight Systems trace. 

This invocation became the primary kernel used for the bottleneck diagnosis.

---

# 14. GPU Speed-of-Light Analysis

The most important high-level utilization metrics were:

```text
Compute (SM) Throughput: 28.09%
Memory Throughput:       14.49%
DRAM Throughput:         14.49%
L1/TEX Throughput:       23.04%
L2 Throughput:            9.80%
```



These values immediately rule out two simple explanations.

## 14.1 Not compute-saturated

If the kernel were primarily constrained by peak arithmetic throughput, we would expect SM throughput to approach the compute capability of the device much more closely.

Instead:

```text
Compute throughput ≈ 28%
```

A large fraction of potential execution capacity is idle.

---

## 14.2 Not DRAM-bandwidth saturated

Likewise:

```text
DRAM throughput ≈ 14.5%
```

is far below peak bandwidth utilization.

Therefore the kernel cannot reasonably be described simply as DRAM bandwidth-bound in this measured execution regime.

This result is worth emphasizing because quantized inference is often casually characterized as memory-bound.

That characterization can be true for many shapes and systems.

It was not supported by this particular kernel measurement.

---

# 15. Scheduler Statistics

The scheduler metrics were much more revealing.

Nsight Compute reported:

```text
One or More Eligible:        17.99%
No Eligible:                 82.01%
Issued Warp / Scheduler:      0.18
Active Warps / Scheduler:     2.00
Eligible Warps / Scheduler:   0.26
```



The key metric is:

```text
No Eligible = 82.01%
```

This means that during approximately 82% of scheduler cycles, there was no active warp ready to issue its next instruction.

The GPU therefore had execution resources available but insufficient ready work to feed those resources.

Nsight Compute also described the issue rate as approximately one instruction issue every 5.6 scheduler cycles in this workload. 

This strongly shifts the diagnosis toward **latency hiding and concurrency**.

---

# 16. Why Eligible Warps Matter

GPU latency hiding depends on having enough independent work resident on each streaming multiprocessor.

A simplified example:

```text
Warp A encounters dependency
        ↓
scheduler selects Warp B

Warp B stalls on memory
        ↓
scheduler selects Warp C

Warp C waits at synchronization
        ↓
scheduler selects Warp D
```

If enough warps are resident and ready, the SM can continue executing useful work even while individual warps stall.

But if the kernel only has a small number of resident warps:

```text
Warp A stalled
Warp B stalled
Warp C stalled
...
no other warp ready
```

the scheduler has nothing to issue.

The hardware then appears underutilized even though neither peak compute nor memory bandwidth has been exhausted.

That pattern matches the measured Marlin kernel:

```text
Low SM throughput
+
Low DRAM throughput
+
Very high No Eligible
```

---

# 17. Occupancy and Resource Pressure

The kernel's launch characteristics help explain why so few warps were available.

The representative Marlin invocation used a block size of:

```text
256 threads
```

and consumed substantial on-chip resources.

The captured configuration showed:

```text
Registers per thread:              198
Dynamic shared memory per block:   65.54 KB
```

The resulting occupancy was approximately:

```text
Theoretical occupancy: 25%
Achieved occupancy:    26.06%
```

with the resource limits restricting residency.

The result is approximately one resident block per SM in this configuration.

With 256 threads per block:

```text
256 threads / 32 threads per warp = 8 warps
```

so one block corresponds to approximately eight resident warps per SM.

That is substantially below the maximum amount of warp-level concurrency the hardware can support.

---

# 18. Bottleneck Interpretation

The measured behavior can be summarized as the following causal chain:

```text
high register footprint
        +
high shared-memory footprint
        ↓
limited blocks resident per SM
        ↓
low warp residency
        ↓
~25% occupancy
        ↓
very few eligible warps
        ↓
scheduler frequently has nothing ready
        ↓
poor latency hiding
        ↓
compute units underutilized
and
memory bandwidth underutilized
```

Therefore the current bottleneck classification is:

> **The profiled steady-state Marlin decode kernel on Tesla T4 is primarily constrained by low residency / insufficient latency hiding rather than by peak DRAM bandwidth or peak compute throughput.**

This is a more actionable conclusion than simply stating that “Marlin is slow.”

It suggests that potentially useful optimization directions involve the kernel's execution structure and resource footprint.

---

# 19. Why “Increase Occupancy” Is Not the Optimization Goal

It would be easy to conclude:

> Occupancy is 25%, therefore occupancy must be increased.

That is too simplistic.

Higher occupancy is not automatically faster.

For example, reducing register usage can force spilling into local memory, which may make a kernel slower even if theoretical occupancy rises.

Likewise, changing tile sizes to reduce shared-memory usage may:

* increase global-memory transactions,
* decrease data reuse,
* reduce Tensor Core efficiency,
* increase instruction count,
* or introduce additional synchronization.

Therefore occupancy should be treated as a **mechanism**, not as the final objective.

The actual objective is:

```text
lower decode latency
```

A successful kernel modification would ideally produce a chain such as:

```text
resource footprint reduced
        ↓
more resident work
        ↓
more eligible warps
        ↓
fewer scheduler idle cycles
        ↓
lower kernel duration
        ↓
lower TPOT
```

All stages need validation.

---

# 20. Shared-Memory Behavior

An earlier full Nsight Compute analysis also exposed non-trivial shared-memory bank conflicts.

That observation is relevant because this kernel already uses a large shared-memory allocation.

Bank conflicts can serialize accesses that would otherwise proceed in parallel.

However, shared-memory conflicts are currently considered a **secondary optimization signal**, not the primary diagnosis.

The strongest evidence remains:

* limited occupancy,
* high resource consumption,
* very few eligible warps,
* and low utilization of both compute and memory bandwidth.

Any future shared-memory optimization would need to demonstrate that reducing conflicts actually improves total kernel and decode latency.

---

# 21. Experimental Optimization: Marlin Atomic Add

During runtime initialization, vLLM emitted an explicit suggestion:

```text
Marlin kernel can achieve better performance for small size_n
with experimental use_atomic_add feature.
```



This made:

```text
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

a reasonable first optimization experiment.

Importantly, the optimization was not chosen because it was expected to produce an attractive benchmark result.

It was chosen because:

1. Marlin had been identified as a major decode cost.
2. vLLM exposed an alternate Marlin execution mode.
3. The optimization could be tested with a controlled A/B experiment.
4. It could be validated with both serving metrics and kernel traces.

---

# 22. Atomic-Add A/B Methodology

Two server configurations were tested.

## Baseline

```bash
unset VLLM_MARLIN_USE_ATOMIC_ADD
```

The notebook explicitly launched the baseline vLLM server with the atomic-add environment variable unset. 

## Experimental configuration

```bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
```

The same serving setup was then repeated with atomic-add enabled. 

The benchmark workload was held constant:

```text
Input length:        32
Output length:       256
Requests:            20
Warmups:              3
Max concurrency:      1
Temperature:          0
```

The primary metrics were:

* TPOT — time per output token,
* ITL — inter-token latency,
* output-token throughput.

For this experiment, TPOT is more relevant than TTFT because the optimization target is the repeated decode path.

---

# 23. End-to-End Baseline

The baseline serving run produced:

```text
Benchmark duration:          24.13 s
Output token throughput:    212.19 tok/s
Mean TTFT:                   29.62 ms
Mean TPOT:                    4.61 ms
Mean ITL:                     4.61 ms
```



This became the reference measurement for the atomic-add experiment.

---

# 24. End-to-End Atomic-Add Result

With atomic-add enabled, the same benchmark reported:

```text
Benchmark duration:          25.46 s
Output token throughput:    201.06 tok/s
Mean TTFT:                   27.87 ms
Mean TPOT:                    4.88 ms
Mean ITL:                     4.88 ms
```



Comparing the decode-focused metrics:

```text
TPOT:
4.61 ms → 4.88 ms

Output throughput:
212.19 tok/s → 201.06 tok/s
```

The experimental configuration therefore produced approximately:

```text
TPOT regression ≈ 5.9%
Throughput regression ≈ 5.2%
```

in this initial end-to-end test.

TTFT improved slightly, but TTFT was not the primary target of this decode-focused experiment.

---

# 25. Kernel-Level Atomic-Add Comparison

Nsight Systems traces were also captured for the baseline and atomic configurations.

At first glance, the atomic trace appeared encouraging.

Several Marlin kernel totals were lower.

However, a closer comparison showed that **many unrelated kernels also became faster in the same trace**.

The comparison included approximately:

| Kernel            | Baseline |   Atomic |
| ----------------- | -------: | -------: |
| `gemvx`           | 379.1 ms | 372.7 ms |
| Marlin #1         | 294.6 ms | 277.7 ms |
| Marlin #2         | 236.2 ms | 219.8 ms |
| Attention         | 200.5 ms | 185.9 ms |
| `reduce_segments` |  25.8 ms |  23.9 ms |

The key problem is causal attribution.

If atomic-add were directly responsible for a 6–7% Marlin improvement, it would not explain why the unrelated attention kernel also became approximately 7% faster in the same run.

That pattern suggests run-to-run factors such as:

* GPU clock variation,
* thermal state,
* power state,
* cloud-host variability,
* or general runtime noise.

The trace therefore cannot be used to claim a clean Marlin speedup.

---

# 26. Why the Atomic Experiment Was Rejected

Performance optimization should ultimately improve the metric the application cares about.

For this experiment, that metric was steady-state decode TPOT.

The evidence was:

```text
Kernel-level:
some Marlin durations appeared lower,
but unrelated kernels also improved.

End-to-end:
TPOT became worse.
```

Therefore the defensible conclusion is:

> **The atomic-add configuration did not produce a validated end-to-end optimization for this workload.**

The experiment was abandoned.

No attempt was made to select only favorable measurements or present an ambiguous microbenchmark result as an application-level speedup.

This is an important part of the methodology:

```text
local kernel improvement
        ≠
guaranteed system improvement
```

A low-level change should be accepted only after it survives end-to-end validation.

---

# 27. What the Atomic Experiment Still Demonstrated

Although it did not produce the desired speedup, the experiment was still useful.

It demonstrated the complete optimization-validation loop:

```text
profile
   ↓
identify expensive subsystem
   ↓
select relevant runtime/kernel change
   ↓
benchmark
   ↓
re-profile
   ↓
compare local and system behavior
   ↓
reject unsupported hypothesis
```

A failed optimization is useful when it narrows the search space.

The alternative would be repeatedly tuning configuration parameters until one benchmark happened to produce a favorable number, which would provide much weaker evidence of actual performance understanding.

---

# 28. Current State of the Investigation

The project can currently be divided into two phases.

## Phase 1 — Profiling and bottleneck diagnosis

**Completed.**

The evidence establishes that:

1. GPTQ INT4 linear layers are executed through Marlin.
2. Marlin is a substantial component of steady-state decode GPU time.
3. A representative Marlin decode kernel does not saturate compute throughput.
4. It also does not saturate DRAM bandwidth.
5. Register and shared-memory requirements restrict residency.
6. Occupancy is approximately 25–26%.
7. Only a small number of warps are eligible to issue.
8. The scheduler has no eligible warp during approximately 82% of cycles.
9. The resulting behavior is consistent with insufficient latency hiding.

---

## Phase 2 — Optimization

**Open.**

One candidate was evaluated:

```text
Marlin atomic-add
```

and rejected after end-to-end validation.

A different optimization direction should now be chosen based on the profiling evidence.

---

# 29. Potential Next Optimization Directions

## 29.1 Marlin tile and resource configuration

The most direct continuation is to examine the relationship among:

* tile size,
* warp count,
* register allocation,
* shared-memory consumption,
* and block residency.

A useful experiment would try to reduce resource pressure enough to permit additional resident work without introducing a more expensive side effect.

The hypothesis would be:

```text
smaller or differently structured tile
        ↓
lower registers/shared memory
        ↓
greater residency
        ↓
more eligible warps
        ↓
better latency hiding
```

The crucial validation metrics would be:

* kernel duration,
* registers/thread,
* shared memory/block,
* occupancy,
* `No Eligible`,
* TPOT.

---

## 29.2 Alternative quantized GEMM backend

Another useful experiment would hold constant:

```text
model
weights
quantization
GPU
workload
```

while changing the quantized linear implementation.

Instead of simply reporting which backend is faster, the goal would be to compare why their behavior differs.

For each backend:

```text
kernel duration
register pressure
shared-memory usage
occupancy
eligible warps
SM utilization
memory utilization
```

could be compared.

That would turn a simple backend benchmark into a kernel-architecture investigation.

---

## 29.3 Small custom W4A16 kernel

A more implementation-focused direction would extract one representative model linear shape and implement a simplified:

```text
FP16 activation × INT4 packed weights
```

kernel using CUDA or Triton.

The purpose would initially not be to outperform Marlin.

The experiment could instead explore the effects of:

* tile size,
* packed-weight layout,
* dequantization placement,
* shared-memory staging,
* number of warps,
* and register usage.

This would provide a direct connection between low-level implementation choices and the profiler metrics observed in the production-quality Marlin kernel.

---

## 29.4 Investigate the `gemvx` path

The largest individual kernel family in the steady-state trace remains unexplained.

It consumes roughly 28% of GPU kernel time and executes once per generated token in the measured workload.

Before attempting aggressive Marlin modification, it may be valuable to identify precisely which model operation maps to this kernel.

If Marlin were substantially accelerated, `gemvx` could become an even more dominant Amdahl's-law limit on end-to-end decode performance.

This is therefore an important system-level follow-up.

---

# 30. Measurement Lessons

This project exposed several practical lessons about GPU performance analysis.

## 30.1 Profile the correct execution region

The first Nsight Systems trace was not invalid.

It simply measured too much.

Startup, compilation, graph capture, and warmup can dominate aggregate statistics and obscure steady-state inference.

The profiling range must correspond to the actual performance question.

---

## 30.2 Kernel identity alone is not enough

The first Nsight Compute capture really was a Marlin kernel.

But it was not the Marlin invocation relevant to the target steady-state decode path.

Repeated kernels can execute with different:

* shapes,
* resource configurations,
* durations,
* and runtime contexts.

A kernel name match alone does not guarantee representative profiling.

---

## 30.3 Low precision does not imply bandwidth saturation

The measured INT4 Marlin kernel used compressed weights, but DRAM utilization was still only approximately 14%.

The bottleneck was more closely associated with scheduler readiness and occupancy.

Optimization decisions should therefore follow measurements rather than generic assumptions about quantized inference.

---

## 30.4 GPU utilization metrics need context

Statements such as:

```text
GPU utilization is high
```

or:

```text
memory bandwidth is low
```

are insufficient by themselves.

The useful interpretation came from combining:

```text
SM throughput
+
DRAM throughput
+
occupancy
+
register usage
+
shared-memory usage
+
eligible-warps statistics
```

The metrics become useful when they support a causal model.

---

## 30.5 Nsight Compute timing is not an application benchmark

Nsight Compute may replay kernels multiple times to collect hardware counters.

Therefore kernel durations measured under full NCU instrumentation should not automatically be compared with normal Nsight Systems durations or end-to-end inference timings.

NCU was used here primarily for **hardware behavior**, not for application-level latency measurement.

---

## 30.6 Eager profiling changes the runtime

Some diagnostic runs used:

```text
--enforce-eager
```

to simplify kernel profiling.

This disables normal CUDA Graph and compilation behavior.

Therefore those runs should not be used to claim production-serving latency.

The application benchmark and system-level steady-state traces use the normal optimized runtime path.

---

# 31. Limitations

Several limitations should be considered when interpreting the results.

## Hardware specificity

The Tesla T4 is a Turing-generation GPU.

Marlin behavior may be materially different on:

* A10,
* A100,
* L4,
* L40S,
* H100,
* H200,
* B200,
* or other architectures.

Kernel tile selection, Tensor Core capabilities, memory hierarchy, register limits, and shared-memory limits vary by GPU architecture.

---

## Model size

Qwen3-0.6B is a small language model.

Larger models can shift the balance among:

* quantized linear operations,
* attention,
* memory traffic,
* KV-cache behavior,
* and serving overhead.

The experiment should therefore be interpreted as a kernel/inference study rather than a universal statement about all GPTQ LLMs.

---

## Batch size

The primary experiment uses batch size / concurrency close to one.

This is particularly relevant for latency-sensitive decode.

Higher batching may substantially alter:

* GEMM dimensions,
* arithmetic intensity,
* kernel selection,
* occupancy,
* and throughput characteristics.

---

## Cloud GPU variability

The experiments were performed in a cloud notebook environment.

Factors such as:

* GPU clock state,
* host load,
* power management,
* and VM-level variability

can influence wall-clock comparisons.

This was particularly visible during the atomic-add experiment.

For future optimization claims, repeated alternating A/B measurements or controlled GPU clock conditions would strengthen the evidence.

---

## No final optimization claim

The project currently contains a diagnosis, not a completed speedup result.

This is intentional.

The final optimization should not be chosen in advance of profiling evidence.

---

# 32. Reproducibility Philosophy

The notebook preserves unsuccessful and corrected profiling attempts rather than deleting them.

This is deliberate.

The sequence:

```text
broad profile
→ identify contamination
→ isolate steady state
→ first kernel capture
→ identify unrepresentative invocation
→ capture later kernel
→ establish diagnosis
```

documents how the final conclusion was reached.

For performance engineering, this history is useful because it shows which measurements should **not** be trusted and why.

The repository therefore separates:

* raw experimental workflow,
* interpreted results,
* and final conclusions.

---

# 33. Current Technical Conclusion

For the measured configuration:

```text
Model:        Qwen3-0.6B GPTQ INT4
Runtime:      vLLM 0.29.0
Backend:      Marlin
GPU:          NVIDIA Tesla T4
Workload:     low-batch autoregressive decode
```

a representative steady-state Marlin kernel exhibited:

```text
Compute throughput             28.09%
Memory throughput              14.49%

Registers per thread           198
Dynamic shared memory/block    65.54 KB

Theoretical occupancy          25%
Achieved occupancy             26.06%

Active warps/scheduler         2.00
Eligible warps/scheduler       0.26
No eligible warp               82.01%
```



The central diagnosis is therefore:

> **The measured Marlin decode kernel is not primarily constrained by peak compute throughput or DRAM bandwidth. High on-chip resource usage limits residency, leaving too few eligible warps to hide execution latency effectively.**

That diagnosis is the main completed result of the current project.

---

# 34. Next Question

The profiling stage answered:

> Why is this Marlin decode kernel underutilizing the T4?

The next stage should answer:

> **Can the compressed linear execution path be changed so that more useful work remains eligible for execution, reducing kernel time and ultimately improving end-to-end TPOT?**

A successful next experiment should demonstrate improvement across multiple layers of evidence:

```text
resource behavior
        ↓
scheduler behavior
        ↓
kernel duration
        ↓
model decode
        ↓
end-to-end TPOT
```

Until that chain is demonstrated, the optimization result remains open.

---

# Conclusion

This investigation started as a simple question about compressed-model inference performance but evolved into a more specific hardware-level diagnosis.

Nsight Systems first established where steady-state decode GPU time was going.

It showed that Marlin quantized linear kernels constituted a major fraction of GPU execution, alongside a large `gemvx` operation and attention. 

Nsight Compute then showed why at least one representative Marlin decode kernel was not using the device efficiently.

The kernel did not saturate arithmetic throughput.

It did not saturate DRAM bandwidth.

Instead, its register and shared-memory footprint restricted occupancy and warp residency, while the scheduler reported no eligible warp for approximately 82% of cycles. 

An initial optimization candidate, Marlin atomic-add, was then subjected to both kernel-level and end-to-end validation. The end-to-end benchmark regressed from **4.61 ms to 4.88 ms TPOT** and from **212.19 to 201.06 output tokens/s**, so that path was not adopted.  

The project therefore ends its current phase with a bottleneck diagnosis rather than a manufactured speedup claim.

The profiling work is complete enough to define the next engineering problem:

> **Reduce the constraints on the quantized decode execution path in a way that improves not only a local kernel metric, but the actual end-to-end generation latency.**

I would put this at `docs/profiling-analysis.md` and keep the README much shorter, linking to it as **“Detailed profiling methodology and analysis.”**
