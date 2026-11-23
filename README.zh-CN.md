# SQD-GPU-Metal

[English](README.md)

面向 Apple Silicon 的原生 **MLX/Metal Selected Basis Diagonalization（SBD）** 核心，用 GPU 加速 SQD 中稀疏、行列式结构的 selected-CI 本征求解。

本仓库是 **TyxonQ 1.5.0 Metal 原生实现的独立展示版**：代码从 TyxonQ 的 `src/tyxonq/numerics/sbd/metal/` 导出，仅将异常类型和内核资源路径改为独立包形式，不简化 H@C、Davidson、S² 或 RDM 内核。

## 已集成到 TyxonQ 1.5.0

**这部分 feature 已完整集成到 TyxonQ 1.5.0。实际开展 SQD 计算时，直接使用 TyxonQ 即可，不需要单独安装本仓库。**

```bash
python -m pip install 'tyxonq[metal]'
```

```python
from tyxonq.applications.chem.algorithms.sqd.gpu import solve_sci_gpu

result, report = solve_sci_gpu(
    ci_strings, h1e, eri, norb, nelec,
    backend="metal",       # "auto" 按 CUDA SBD → Metal SBD → PyTorch CPU 选择
    rdm_backend="cpu",    # 默认；可选 "metal" 加速异自旋 2-RDM
)
```

TyxonQ 主项目：<https://github.com/QureGenAI-Biotech/TyxonQ>

## 为什么需要原生 Metal SBD

SQD 的 selected-CI 步骤不是普通稠密矩阵乘法。Hamiltonian 只在满足费米子激发规则的行列式之间连接，并包含 alpha/beta 同自旋与异自旋项、轨道地址映射和符号。通用 MPS/MLX 数组迁移无法自动解决这些稀疏结构和不规则访存。

本项目把长向量和高频数值核放到 Apple GPU，同时保留 PySCF 的连接表、化学积分和独立 FP64 验收边界。

## 完整实现与简单结构

```text
sqd-gpu-metal/
├── sqd_gpu_metal/
│   ├── __init__.py       # 公共 API
│   ├── solver.py         # H@C、S²、Davidson、Metal 执行与精度边界
│   ├── rdm.py            # 可选异自旋 2-RDM Q60 内核接口
│   └── kernels.metal     # 全部自定义 MSL 内核
├── example.py            # H2 求解 + PySCF 独立验收
├── pyproject.toml        # 独立安装与内核打包
├── LICENSE               # Apache-2.0
├── README.md             # 英文
└── README.zh-CN.md       # 简体中文
```

核心代码只有 `solver.py`、`rdm.py` 和 `kernels.metal` 三个文件，没有编译器、设备层或分布式运行时等多层架构。

### 已实现范围

- alpha same-spin、beta same-spin 和 mixed-spin 的完整 H@C；
- connected-rows 与 pair-table 两类数据路径及自动选择；
- 原生 S² 跃迁、轨道交换、费米子符号和散射；
- MLX 常驻系数、算符像和 Davidson 长向量；
- 两遍 Gram–Schmidt、thick restart、投影缓存和独立最终残差；
- 确定性 FP32 长内积与归约；
- 可选异自旋 2-RDM 的 FP32 双分量补偿乘积和 Q60 整数归约；
- 显式 `mx.stream(mx.gpu)` 和 `mx.eval` 完成边界及真实原生调用计数。

## 独立运行

要求：macOS、Apple Silicon、Python 3.10+。MLX 只在支持的环境中安装；其他平台不能执行 Metal 内核。

```bash
cd sqd-gpu-metal
python -m pip install -e .
python example.py
```

示例构造 H2/STO-3G 的完整 selected-CI 空间，在 Metal 上运行 Davidson，并使用原始 PySCF 算符独立检查能量、最终残差和 Q60 异自旋 2-RDM。当前实测能量误差约 `5.44e-9 Ha`、残差约 `8.16e-9`、RDM 最大误差约 `4.44e-16`。若 Metal 不可用或数值验收失败，示例会显式报错，不会静默切换到 CPU。

也可以直接使用核心 API：

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

连接表由 `pyscf.fci.selected_ci._all_linkstr_index` 构造；完整调用过程见 `example.py`。

## 实测性能

以下结果来自 TyxonQ 1.5.0 合并 PR #21 后的同一份 Metal 实现，不是合成预估或仅内核加速比。

测试协议：Apple Silicon、MLX 0.32.4、PyTorch 2.8.0、PySCF 2.14.0；H2O/NH3/CH4，6-31g(d,p)，CAS(4e,12o–18o)，4,356–23,409 determinants。每个点分别执行 Metal 和 PyTorch CPU，1 次冷启动加 5 次稳态，并用两个独立进程复测。所有 36 组配对负载均正常收敛并通过残差/RDM 验收。

**加速倍率 = CPU 整体中位墙钟 / Metal 整体中位墙钟**，包含 setup、正常收敛 solver、CPU RDM 和 validation；越大越好。

| 分子 | 23,409 determinants 第一轮 | 第二轮 |
| --- | ---: | ---: |
| H2O | **2.57×** | **2.67×** |
| NH3 | **1.98×** | **2.19×** |
| CH4 | **2.18×** | **2.24×** |

- 三种分子最早受测的双轮稳定整体交叉点均为 **8,281 determinants（CAS(4e,14o)）**。
- 最大受测规模下，Metal 整体执行最高达到 **2.67× CPU 速度**。
- 4,356 determinants 第一轮仍偏向 CPU，因此不把小规模单轮收益包装成稳定交叉点。
- 固定 40-cycle FP32 profile 与正常收敛结果分开测量；固定循环不用于宣称基态能量收敛。

这些数字是单次固定 CAS 的 selected-CI 整体结果，不外推到完整迭代 SQD 采样/恢复、任意活性空间、其他硬件或未实测 CUDA。

## 主要创新点与工程难点

1. **面向行列式连接关系，而非通用稠密 GPU 移植**  
   connected-rows 只访问相连 determinant；pair-table 在同自旋连接矩阵过大时限制 setup/内存开销。难点是保持 PySCF 的地址、chunk、费米子符号和临时张量布局完全一致。

2. **Metal 上完成真实求解主循环**  
   H@C、S²、正交化、残差和长向量保留在 GPU，CPU 只负责连接表和小投影 Hermitian FP64 `eigh`。必须显式处理 MLX 异步执行，否则计时和结果都可能失真。

3. **确定性 FP32，而不是隐式降精度 GEMM**  
   MLX 0.32.4 在部分 Apple 硬件上可能对 FP32 `mx.matmul` 使用 NAX 降精度路径。生产实现使用自定义确定性 FP32 归约，不读取或修改进程级 `MLX_ENABLE_TF32`。研究完整 FP32 GEMM 时，必须在进程启动前设置 `MLX_ENABLE_TF32=0`。FP32 内核不等于端到端 FP64。

4. **用独立数值验收约束性能优化**  
   H@C/S² golden、最终新残差、正常收敛、逐自旋 RDM、粒子数和能量分别验证。CSR scattering、四输出 tile 和默认降精度 GEMM 等候选不能仅凭局部耗时采用。

5. **补偿 Q60 异自旋 RDM**  
   高低两个 FP32 分量恢复系数乘积的一阶误差，再以 Q60 整数归约降低累加顺序差异。实现包含归一化检查、规模上限和超限错误；它只覆盖异自旋块，不代表全自旋块原生 RDM。

## 数值与功能边界

- 当前是**单 Apple GPU**实现，不是多 GPU、多机或 MPI/NCCL 版本。
- 多 Apple GPU 的后续路线是 MLX 原生 `mx.distributed`/`mlx.launch`，但尚未在本核心中实现。
- RDM 默认在 CPU；`mixed_rdm_q60` 仅加速异自旋 2-RDM，其他块仍由 CPU FP64 路径负责。
- 环境不可用和数值失败是不同事件；本核心不会用 CPU 结果掩盖 Metal 内核错误。
- 完整 SQD 工作流、后端自动选择、target-spin、持久 CPU validator 和批量接口请使用 TyxonQ 1.5.0。

## 算法与代码参考

- **TyxonQ**：本独立实现的直接来源，功能已集成到 1.5.0。  
  <https://github.com/QureGenAI-Biotech/TyxonQ>
- **RIKEN R-CCS SBD**：Selected Basis Diagonalization 核心算法与 TPB/CUDA 参考实现；固定参考提交 `93ebabec6c841d0ade706018b053795d8c1db9bd`。  
  <https://github.com/r-ccs-cms/sbd>
- **Qiskit SBD eigensolver Python wrapper**：Python/PySCF 接口参考；固定参考提交 `b49c22b54813be1e0344ba52f8ffd03e3825433e`。  
  <https://github.com/Qiskit/sbd-eigensolver-python>
- **Apple MLX**：Metal 数组、流、JIT 自定义内核与执行环境。  
  <https://github.com/ml-explore/mlx>
- **PySCF**：化学积分、行列式连接表和独立 CPU FP64 数值参考。  
  <https://github.com/pyscf/pyscf>

SBD GPU 算法参考论文：**arXiv:2601.16637**。本仓库不是上述项目的官方替代品；引用和复用关系以各自许可证与固定提交为准。

## License

Apache License 2.0，见 `LICENSE`。
