# SQD-GPU-Metal

[简体中文](README.zh-CN.md)

A native **MLX/Metal Selected Basis Diagonalization (SBD)** core for Apple Silicon, accelerating the sparse, determinant-structured selected-CI eigensolver in SQD on the GPU.

This repository is a **standalone showcase of the native implementation in TyxonQ 1.5.0**: the code is exported from TyxonQ's `src/tyxonq/numerics/sbd/metal/`, with only the exception types and kernel resource paths changed to standalone-package form. The H@C, Davidson, S² and RDM kernels are not simplified.

## Integrated into TyxonQ 1.5.0

**This feature has been fully integrated into TyxonQ 1.5.0. For actual SQD calculations, use TyxonQ directly; installing this repository separately is not required.**

```bash
python -m pip install 'tyxonq[metal]'
```

```python
from tyxonq.applications.chem.algorithms.sqd.gpu import solve_sci_gpu

result, report = solve_sci_gpu(
    ci_strings, h1e, eri, norb, nelec,
    backend="metal",       # "auto" selects CUDA SBD → Metal SBD → PyTorch CPU
    rdm_backend="cpu",    # Default; "metal" can accelerate the opposite-spin 2-RDM
)
```

TyxonQ main project: <https://github.com/QureGenAI-Biotech/TyxonQ>

## Why a native Metal SBD is needed

The selected-CI step in SQD is not an ordinary dense matrix multiplication. The Hamiltonian connects only determinants that satisfy fermionic excitation rules, and includes alpha/beta same-spin and opposite-spin terms, orbital address mapping, and signs. Generic MPS/MLX array migration cannot automatically resolve this sparse structure and irregular memory access.

This project puts long vectors and high-frequency numerical kernels on the Apple GPU, while retaining PySCF's connectivity tables, chemical integrals, and an independent FP64 validation boundary.

## Complete implementation, simple structure

```text
sqd-gpu-metal/
├── sqd_gpu_metal/
│   ├── __init__.py       # Public API
│   ├── solver.py         # H@C, S², Davidson, Metal execution and precision boundary
│   ├── rdm.py            # Optional opposite-spin 2-RDM Q60 kernel interface
│   └── kernels.metal     # All custom MSL kernels
├── example.py            # H2 solve + independent PySCF validation
├── pyproject.toml        # Standalone installation and kernel packaging
├── LICENSE               # Apache-2.0
├── README.md             # English
└── README.zh-CN.md       # Simplified Chinese
```

The core consists of only `solver.py`, `rdm.py`, and `kernels.metal`. There is no multi-layer architecture such as a compiler, device layer, or distributed runtime.

### Implemented scope

- Full H@C for alpha same-spin, beta same-spin, and mixed-spin terms;
- Two data paths, `connected-rows` and `pair-table`, with automatic selection;
- Native S² transitions, orbital swaps, fermionic signs, and scattering;
- MLX-resident coefficients, operator images, and Davidson long vectors;
- Two-pass Gram–Schmidt, thick restart, projection cache, and independent final residual;
- Deterministic FP32 long dot products and reductions;
- Optional opposite-spin 2-RDM with an FP32 two-component compensated product and Q60 integer reduction;
- Explicit `mx.stream(mx.gpu)` and `mx.eval` completion boundaries, plus real native call counts.

## Standalone run

Requirements: macOS, Apple Silicon, Python 3.10+. MLX is installed only in supported environments; Metal kernels cannot execute on other platforms.

```bash
cd sqd-gpu-metal
python -m pip install -e .
python example.py
```

The example builds the full selected-CI space for H2/STO-3G, runs Davidson on Metal, and independently checks the energy, final residual, and Q60 opposite-spin 2-RDM with raw PySCF operators. Current measured errors are approximately `5.44e-9 Ha` in energy, `8.16e-9` in the residual, and `4.44e-16` maximum error in the RDM. If Metal is unavailable or numerical validation fails, the example raises an explicit error rather than silently falling back to CPU.

The core API can also be used directly:

```python
from sqd_gpu_metal import (
    MetalSelectedCIContractor,
    MetalSelectedS2Contractor,
    absorb_h1e,
    davidson_lowest,
    make_hdiag,
    mixed_rdm_q60,
)
```

The connectivity table is constructed with `pyscf.fci.selected_ci._all_linkstr_index`; see `example.py` for the full call flow.

## Measured performance

The following results come from the same Metal implementation after TyxonQ 1.5.0 merged PR #21. They are neither synthetic estimates nor kernel-only speedups.

Test protocol: Apple Silicon, MLX 0.32.4, PyTorch 2.8.0, PySCF 2.14.0; H2O/NH3/CH4, 6-31g(d,p), CAS(4e,12o–18o), 4,356–23,409 determinants. At each point, Metal and PyTorch CPU were run separately with one cold-start and five steady-state runs, and re-measured in two independent processes. All 36 paired workloads converged normally and passed residual/RDM validation.

**Speedup = median end-to-end CPU wall-clock / median end-to-end Metal wall-clock**, including setup, a normally converging solver, CPU RDM, and validation; higher is better.

| Molecule | 23,409 determinants, first run | Second run |
| --- | ---: | ---: |
| H2O | **2.57×** | **2.67×** |
| NH3 | **1.98×** | **2.19×** |
| CH4 | **2.18×** | **2.24×** |

- For all three molecules, the earliest tested two-run stable end-to-end crossover was **8,281 determinants (CAS(4e,14o))**.
- At the largest tested scale, end-to-end Metal execution reached up to **2.67× CPU speed**.
- At 4,356 determinants, the first run still favored CPU; small-scale single-run gains are therefore not presented as a stable crossover.
- The fixed 40-cycle FP32 profile is measured separately from normally converging results; fixed-cycle runs are not used to claim ground-state energy convergence.

These numbers are end-to-end selected-CI results for a single fixed CAS. They are not extrapolated to full iterative SQD sampling/recovery, arbitrary active spaces, other hardware, or unmeasured CUDA.

## Key innovations and engineering challenges

1. **Designed for determinant connectivity, not a generic dense GPU port**  
   `connected-rows` accesses only connected determinants, while `pair-table` limits setup and memory overhead when the same-spin connectivity matrix becomes too large. The challenge is to keep PySCF's addressing, chunking, fermionic signs, and temporary tensor layout exactly consistent.

2. **The real solver loop runs on Metal**  
   H@C, S², orthogonalization, residuals, and long vectors remain on the GPU; the CPU only handles connectivity tables and the small projected Hermitian FP64 `eigh`. MLX asynchronous execution must be handled explicitly, otherwise both timings and results can be distorted.

3. **Deterministic FP32, not implicit reduced-precision GEMM**  
   MLX 0.32.4 may use a NAX reduced-precision path for FP32 `mx.matmul` on some Apple hardware. The production implementation uses a custom deterministic FP32 reduction and does not read or modify the process-level `MLX_ENABLE_TF32`. When investigating full FP32 GEMM, set `MLX_ENABLE_TF32=0` before process startup. FP32 kernels are not equivalent to end-to-end FP64.

4. **Independent numerical validation constrains performance optimization**  
   H@C/S² golden values, a freshly computed final residual, normal convergence, per-spin RDMs, particle number, and energy are validated separately. Candidates such as CSR scattering, four-output tiles, and the default reduced-precision GEMM cannot be adopted based on local timings alone.

5. **Compensated Q60 opposite-spin RDM**  
   High and low FP32 components recover the first-order error in the coefficient product, and a Q60 integer reduction reduces differences caused by accumulation order. The implementation includes normalization checks, a size limit, and an over-limit error; it covers only the opposite-spin block and does not represent native RDMs for all spin blocks.

## Numerical and functional boundaries

- This is a **single Apple GPU** implementation, not a multi-GPU, multi-node, or MPI/NCCL version.
- The future route for multiple Apple GPUs is MLX's native `mx.distributed` / `mlx.launch`, but it is not implemented in this core.
- RDMs default to CPU; `mixed_rdm_q60` only accelerates the opposite-spin 2-RDM, while other blocks remain on the CPU FP64 path.
- An unavailable environment and a numerical failure are distinct events; this core does not mask Metal kernel errors with CPU results.
- For the full SQD workflow, backend auto-selection, target-spin, a persistent CPU validator, and batch APIs, use TyxonQ 1.5.0.

## Algorithm and code references

- **TyxonQ**: the direct source of this standalone implementation, with the feature integrated into 1.5.0.  
  <https://github.com/QureGenAI-Biotech/TyxonQ>
- **RIKEN R-CCS SBD**: the Selected Basis Diagonalization algorithm and TPB/CUDA reference implementation; fixed reference commit `93ebabec6c841d0ade706018b053795d8c1db9bd`.  
  <https://github.com/r-ccs-cms/sbd>
- **Qiskit SBD eigensolver Python wrapper**: Python/PySCF interface reference; fixed reference commit `b49c22b54813be1e0344ba52f8ffd03e3825433e`.  
  <https://github.com/Qiskit/sbd-eigensolver-python>
- **Apple MLX**: Metal arrays, streams, JIT custom kernels, and the execution environment.  
  <https://github.com/ml-explore/mlx>
- **PySCF**: chemical integrals, determinant connectivity tables, and an independent CPU FP64 numerical reference.  
  <https://github.com/pyscf/pyscf>

Reference paper for the SBD GPU algorithm: **arXiv:2601.16637**. This repository is not an official replacement for the projects above; citation and reuse follow their respective licenses and fixed commits.

## License

Apache License 2.0, see `LICENSE`.
