# 习题：用 EasyASC 实现昇腾 A3 上的 KDA Chunk Prefill

本题要求基于 EasyASC，为昇腾 A3 实现高效的 Kimi Delta Attention（KDA）**chunk prefill 前向计算**，并研究一个贯穿开发过程的问题：**如何切分、组织和调度多个 kernel，才能让整个算子体系高效运行？**

需要交付两项相互支撑的成果：**一份可复用的切分策略评估模型，以及一份可运行的 A3 kernel 实现。** 先提出策略和预测，再完成实现，用实验检验并修正判断。评估模型可以是 skill、文档、代码或工具；我们希望这些方法能够泛化至其他算子的开发流程中，并完善昇腾算子的工具链与开发 skill。

这里是题目的总体说明。配套的[习题包说明](kda_a3_exercise/README.md)给出具体接口和运行方式，[可执行约束](kda_a3_exercise/contract.json)与[检查器](kda_a3_exercise/check.py)定义正确性规则。当前题包版本为 `0.3.0`；本题要求前向，反向与训练链路属于可选扩展。

## 1. EasyASC 是什么？

EasyASC 是用于编写昇腾 kernel 的 Python DSL。你通过 Python 函数描述设备上的计算、数据搬运、缓冲区和同步，框架将其转换为指令表示，组织 Cube 与 Vector 执行路径，并支持内置 simulator、设备代码生成和设备运行。

**本题使用 cannbot-skills 中的 [EasyASC 社区版本（ops-easyasc-dsl）](https://gitcode.com/cann/cannbot-skills/tree/master/plugins-community/ops-easyasc-dsl)。** 请按该社区入口的说明准备源码和环境。编写纯净 kernel 时，直接以 **EasyASC 源码根目录的 `AGENTS.md`** 作为 AI agent 的入口，再按该文档指引读取后续材料。

你可以从一个小 kernel 入手，理解计算如何映射到硬件管道，再逐步扩展为多 kernel 系统。仓库中的 A5 `kda_fwd`、`kda_bwd` 提供了已有工程实现，可用于研究数学分解、数据流、融合和同步。移植到 A3 时，需要重新评估这些选择。

本习题目录与 EasyASC 源码独立：运行 PyTorch golden 和题包自检只需要题包的 Python 依赖；开发、仿真和运行 EasyASC kernel 时，需要从上述社区入口准备 EasyASC 源码，并将其根目录设为后文的 `EASYASC_ROOT`。

```text
kda_a3_practice/
├── README.md                  # 题目总体说明
├── kda_a3_exercise/            # Golden、接口、用例、检查器和提交模板
│   ├── README.md              # 题包使用说明
│   ├── EVALUATION.md          # 切分策略评估报告模板
│   ├── PROVENANCE.md          # 公开源码来源及适配记录
│   └── ...
└── build_package.py           # 打包题面与完整题包
```

## 2. 如何通过 EasyASC 学习昇腾架构？

建议按照“编程模型 → 数据流 → 同步 → trace → 优化”的顺序阅读和实验。下表的路径均相对于 **EasyASC 仓库根目录**，不在本习题目录内：

| 目的 | 仓库内入口 |
| --- | --- |
| 安装与运行第一个示例 | `README_CN.md`、`doc/01_quickstart.md` |
| 学习编程模型与 kernel 写法 | `doc/02_programming_model.md`、`doc/03_write_your_first_kernel.md` |
| 理解混合流水线与同步 | `doc/04_mixed_pipeline_and_sync.md` |
| 使用 simulator 和 trace | `doc/05_simulator_and_trace.md` |
| 配合 AI agent 开发 | 从 `AGENTS.md` 开始，再按指引进入 `agent/ROUTER.md` |
| 查询 API 与设备约束 | `doc/api/index.md`、`agent/references/facts-device-runtime.md` |
| 查找小型 kernel | `kernels/README.md` |
| 研究 A5 KDA 实现 | `projects/a5/kda_fwd/README.md`、`projects/a5/kda_bwd/README.md` |
| 阅读中文教程 | `doc_cn/index.md` |

学习每个 kernel 时，先画出一个数据块的搬运路径：

```text
Cube：   GM → L1 → L0A/L0B → 矩阵计算 → L0C → 写回
Vector： GM → UB → 向量计算 → UB → GM
```

再回答四个问题：每个值存在哪里、生命周期有多长？每个操作由哪个管道执行？消费者何时可以开始、缓冲区何时可以复用？batch、head、chunk 和 tile 如何分配到核心？这些判断会决定分块、资源占用、同步和可用并行度。

### A3 上需要重新评估的地方

提交的 kernel 使用 `from easyasc.a3 import *`。A3 使用 C220 API，SoC 配置与编译目标应按实际设备选择。A5 使用 C310 API，其 `@vf` 和寄存器级 micro 代码需要改用 A3 的公开 API 实现。一个 Python 进程只导入一个目标设备入口，A5 参考实验与 A3 实现应在不同进程中运行。

当前 A3 的 Cube/Vector 交接通过 GM workspace，不能直接沿用 A5 的 `L0C → UB`、`UB → L1` 片上通路。片上缓冲区容量和跨侧同步也需要按 A3 重新检查；C220 的跨侧等待涉及 `Pipe.S`，等待位置会影响后续工作能否并行推进。可结合仓库的 `agent/references/patterns/a2-mixed-pipeline.md` 研究这类数据流。

**GM 桥接不等于每次都访问 HBM。** 分块与调度如果能让生产者和消费者的数据在 L2 中及时衔接，GM 桥接仍可能获得很高的有效带宽。这也是本题评估切分策略时需要研究的问题。

## 3. 如何上手 kernel、观察 pipe 并行并优化？

### 使用 AI agent 与 EasyASC 配合

**编写纯净 kernel 时，直接从 EasyASC 源码根目录的 `AGENTS.md` 开始。** 它提供仓库级协作约定，并引导 agent 通过 `agent/ROUTER.md` 选择具体任务的工作流。让 agent 能够访问 EasyASC 源码和本习题目录，并明确告知两个目录的实际路径；如果工具不会自动加载 `AGENTS.md`，请显式指定该文件的绝对路径，要求 agent 先读取它。

获取仓库约定后，具体任务按以下顺序展开。这里的路径均相对于 EasyASC 仓库根目录：

```text
agent/ROUTER.md                      选择当前任务对应的一个工作流
  → agent/common-language.md         完整阅读，统一架构与 DSL 术语
  → router 选定的 playbook           按当前阶段开展工作
  → 按需查阅设备事实、模式与示例       为实现决策提供依据
```

进入 kernel 编写阶段时，再按工作流读取 `agent/references/authoring-preflight.md`。让 agent 通过索引和示例选择器查找相关材料，围绕当前问题补充上下文。

对本题，可以先让 agent 阅读 golden 和接口，开展多 kernel 切分评估；切分明确后再实现各阶段。后续出现数值错误时选择调试工作流，正确性通过后选择优化工作流。每次给出当前目标、相关文件、已通过的用例和待解决的问题，由 router 选择对应路线。

下面是一段用于开始切分评估的提示词，将路径占位符替换为实际目录即可：

```text
EasyASC 仓库：<EasyASC 仓库的绝对路径>
习题目录：<kda_a3_practice 的绝对路径>

请读取 EasyASC 根目录的 AGENTS.md，再从 agent/ROUTER.md 选择本阶段的
一个工作流，完整阅读 agent/common-language.md，然后按选定 playbook 工作。

当前任务是昇腾 A3 KDA chunk prefill 前向的多 kernel 切分评估。
请阅读习题 README，以及题包中的 README.md、contract.json、golden.py、
cases.json 和 EVALUATION.md；需要理解分块公式时查阅题包内的上游源码。

请给出输入输出与数学语义、至少两种候选切分的 DAG、各 kernel 的职责、
chunk 内并行与跨 chunk 状态依赖，并估计片上资源、workspace 和 L2 工作集。
区分逻辑 GM 流量、L2 中转与实际 HBM 流量，给出可检验的性能预测，
以及分阶段实现和验证计划。API 与硬件结论请附所依据的仓库文件路径。

遵守题包的 host 白名单、辅助矩阵和 2% 相对误差规则。
性能只统计各 kernel 实际执行耗时之和，启动开销不计入。
开发验证使用 OpExec(..., simulator=True)。
```

实现阶段可以让 agent 每次完成并验证一个阶段，说明其搬运、转换、缓冲区与同步依据；调试时提供失败用例、误差报告和日志；优化时提供 trace 和当前切分，让它先提出瓶颈假设，再用实验比较。你需要检查这些判断是否得到代码、测试和 trace 支持，并将有效的方法整理为可复用的 skill 或工具，作为本题的评估模型交付。

### 先跑通题包和小型 kernel

以下命令从本习题根目录执行。已有可用的 PyTorch 环境时可直接使用；新环境先安装题包依赖：

```bash
python -m pip install -r kda_a3_exercise/requirements.txt
python -m pytest -q kda_a3_exercise/tests
python kda_a3_exercise/check.py --check-golden --suite smoke
```

题包自检无需下载模型权重，也不需要 FLA、Triton、CUDA 或昇腾硬件。`--check-golden` 用于检查参考与评测流程，不能作为 A3 实现通过的证明。

准备 EasyASC 环境时，按仓库的 quickstart 安装相应依赖，并设置仓库路径。下面假设 EasyASC 位于 `~/projects/easyasc`，请按实际位置修改：

```bash
export EASYASC_ROOT="$HOME/projects/easyasc"
export PYTHONPATH="$EASYASC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$EASYASC_ROOT/kernels/a2/matmul/matmul_half_basic.py"
```

这是一个已跟踪的 A2 Cube 示例，可帮助理解 C220 数据流。接下来可以新建使用 A3 入口的小型 kernel，再逐步实现 KDA。实验脚本由你自行编写，也可以使用仓库中已有的示例和测试入口。

### 用 trace 找到等待与重叠

在自己的 `OpExec` 调用中打开 trace。下面是调用示意，`kernel_fn` 和参数需替换为实际实现：

```python
out = OpExec(kernel_fn, out_dir="results/build", simulator=True,
             trace="results/trace.json")(...)
```

也可以先运行仓库已有的 A5 KDA 参考实验，学习如何读取多 kernel 的 trace：

```bash
python "$EASYASC_ROOT/projects/a5/kda_fwd/test_script/module_vs_ref.py" \
  --module compose --mode simulator --B 1 --H 1 --HV 1 --C 2 \
  --K 128 --V 128 --chunk-size 64 --trace \
  --out-dir results/a5_reference

python "$EASYASC_ROOT/tools/analyze_sim_trace.py" \
  results/a5_reference/traces/seed_2026/compose_sub45_fused.json \
  --group-by lane-pipe
```

使用兼容 Chrome/Perfetto 格式的查看器打开 JSON，沿同一核心、同一数据块观察 MTE2、MTE1、M、FIX、V、MTE3 等管道，再比较不同核心的结束时间。重点关注：加载下一个 tile 时能否计算当前 tile；Cube 与 Vector 是否能够并行；空泡来自数据依赖、缓冲区复用、过早等待还是负载不足；增加缓冲槽位是否值得。

修改分块、融合或同步后，同时比较正确性和 kernel 完成时间。报告 pipe 占用率时说明统计区间；更高的局部占用率未必带来更短的完成时间。遇到 `auto_sync` 等警告时，应定位原因并修正实现；若属于框架限制，提交可复现用例和具体修复建议。

### 开发验证与性能测量

开发阶段使用 `OpExec(..., simulator=True)`。EasyASC simulator 帮助验证语义、数据流和调度，其周期由时序模型估计；当前 A3 使用 `a2_cycle_model.json`，并非单独校准的 A3 时序模型。

A3 的设备代码仿真使用 **CAModel**，不支持 `cannsim record` 路径。由于仿真耗时很长，CAModel 在本题中仅用于小规模用例的正确性验证，无需在报告中提交或标注其结果，也不用于性能评估。

需要这类验证时，可以用 `OpExec(..., simulator=False, debug=True, gen_only=True)` 生成调试工程，再将生成的 `<out_dir>_kernel_script/b.sh` 和 `r.sh` 中的 `-r npu` **都改为 `-r sim`**，然后重新编译并运行。`-v` 参数应选择实际的 A3 SoC 配置，例如 `Ascend910_9362`。

`gen_only=True` 只生成产物，编译和仿真需要兼容的 CANN/CAModel 环境。真机运行使用 `simulator=False`，环境与产物说明见 EasyASC 的 `doc/06_codegen_and_runtime.md`。

报告应区分 EasyASC 模拟周期与真机耗时。A3 硬件性能结论需要 A3 真机数据支持。

## 4. KDA 是什么？题包提供了哪些参考与约束？

KDA 是 Kimi Linear 中的递归线性注意力组件，通过逐通道遗忘门控制状态衰减，并用 delta 更新修正状态。背景见 [Kimi Linear 论文](https://arxiv.org/abs/2510.26692)。对一个 batch、一个 value head，设归一化后的 `q_t`、`k_t` 为 K 维向量，`v_t` 为 V 维向量，状态 `S_t` 的形状为 `[K,V]`，其语义可写为：

```text
decayed_state = diag(exp(g_t)) @ S_(t-1)
residual_t    = v_t - k_t^T @ decayed_state
S_t           = decayed_state + beta_t * k_t @ residual_t^T
o_t           = (K**-0.5 * q_t)^T @ S_t
```

本题关注多 token 的 **chunk prefill**：将 chunk 内工作组织为门控前缀和、因果 QK/KK 矩阵、三角求解和 WY 中间量等计算，再生成该 chunk 的输出与下一 chunk 的状态。请说明实现如何利用 chunk 内并行、处理跨 chunk 状态依赖。单 token recurrent decode 循环不属于本题的目标实现。

### 公开模型与 PyTorch golden

题包的来源链是公开模型 [Kimi-Linear-48B-A3B-Instruct 的 KDA 调用](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py)，以及 [Flash Linear Attention 的纯 PyTorch KDA 参考](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py)。上游源码、固定版本、文件哈希、许可证与适配说明保存在题包中，详见 [PROVENANCE.md](kda_a3_exercise/PROVENANCE.md)。Golden 独立于仓内 A5 KDA 项目的参考实现。

[golden.py](kda_a3_exercise/golden.py)按模型调用约定完成 Q/K 归一化后，调用上游 `naive_chunk_kda`。主 golden 默认使用 64 token 的 chunk，并通过逐 chunk 调用控制 CPU 内存占用；recurrent 仅用于独立交叉验证。尾块在 golden 内部作不改变状态的补齐，提交实现收到的仍是原始长度。**实现的内部 chunk 大小、kernel 数量和中间布局由你决定。**

### 输入输出与正确性

本题包含 KDA 核心和 Q/K 归一化。Q/K/V 投影、短卷积、原始门控激活、输出门控归一化及输出投影在题目范围之外。输入为固定种子生成的合成算子数据，无需运行完整模型。

```python
def forward(q, k, v, g, beta, initial_state=None, *, aux=None):
    return o, final_state
```

| 张量 | 连续布局 | dtype 与语义 |
| --- | --- | --- |
| `q`、`k` | `[B,T,H,K]` | bf16，尚未执行 L2 归一化 |
| `v` | `[B,T,HV,V]` | bf16 |
| `g` | `[B,T,HV,K]` | fp32，已激活的自然对数门控增量，未经 cumsum |
| `beta` | `[B,T,HV]` | fp32，已激活的更新权重 |
| `initial_state` | `[B,HV,K,V]` | fp32，`None` 表示零状态 |
| `o` | `[B,T,HV,V]` | bf16 |
| `final_state` | `[B,HV,K,V]` | fp32，始终返回 |

Q/K 归一化的 golden 语义为：先以 fp32 计算 `x / sqrt(sum(x*x) + 1e-6)`，再存为 bf16，进入 chunk KDA 计算。基础题为 `H=HV`、`K=V=128`，需要正确处理尾块与初始状态。

| 用例集 | 内容 |
| --- | --- |
| `smoke` | 3 个基础用例子集，长度为 128、512、513，便于快速调试。 |
| `core` | 16 个基础用例，长度 128–2048，覆盖多 batch/head、尾块、强弱门控、beta 边界、初态和归一化边界。 |
| `extended` | 3 个可选扩展用例，覆盖分组 value head、不等 K/V 和非整齐维度。 |
| `performance` | 2 个模型配置形状：32 head、128 维，长度为 1024 和 4096；检查器仍只验证正确性。 |

对 `o` 和 `final_state`，逐元素误差满足 `abs(actual-golden) <= 0.02 + 0.02*abs(golden)`，且非全零参考的相对 L2 误差不超过 **2%**。阈值参考 A5 前向的精度标准；中间计算和存储精度由你选择。检查器还检查输出形状、dtype、连续性、有限值，以及输入是否被修改。

### Host 边界与辅助矩阵

Host 侧用于组织调用、查询元信息、做不拷贝的 view/reshape 等重解释、通过 `torch.empty` 类接口分配未初始化缓冲区，以及必要的同 dtype CPU/NPU 传输。**归一化、cumsum、矩阵计算、dtype 转换和布局转换等工作均应在 kernel 中完成。** 题包在提交模块导入及执行期间进行基础白名单和源码检查，禁止通过 NumPy、`.item()` 等路径提取数据进行 host 计算。完整规则与检查边界见[题包说明](kda_a3_exercise/README.md)。

评测方会通过只读 `aux` 字典，提供 **32、64、128 阶 bf16** 的下三角 `lower`、上三角 `upper`、全 1 矩阵 `ones` 和单位矩阵 `identity`。上下三角均包含对角线。例如 `aux[64]["lower"]` 可作为 Cube kernel 的输入；对 `[L,K]` 数据 `X`，`lower @ X` 表示沿 token 维的包含当前位置的前缀和，对 `[K,L]` 布局则可使用 `X @ upper`。

辅助矩阵由评测方预先构造，只依赖公开尺寸，不依赖输入数据或 golden。是否使用由你决定；常量构造不计时，kernel 加载、转换和使用矩阵的实际耗时计入评估。输入和辅助矩阵均按只读使用。

### 接入自己的实现

从本习题根目录复制模板并实现 `forward`，确保 EasyASC 已在 `PYTHONPATH` 中：

```bash
cp kda_a3_exercise/submission.py my_solution.py
python kda_a3_exercise/check.py --submission my_solution.py --suite smoke
python kda_a3_exercise/check.py --submission my_solution.py --suite core \
  --report results/correctness.json
python kda_a3_exercise/check.py --submission my_solution.py --suite performance \
  --report results/performance_correctness.json
```

模板未提供 A3 解答，需要先完成实现。基础验收要求整个 `core` 集通过；报告性能的形状也须先通过正确性检查。Host 检查是基础防护，不能单独证明 A3 设备执行或 chunk 内并行，需结合实现源码与运行证据评审。

## 5. 如何评估切分策略与多 kernel 体系？

A5 前向中的门控处理、chunk 内计算、WY 中间量生成、状态更新和输出融合，可以作为理解分解的起点。请根据 A3 的资源与数据通路，重新比较融合边界、分块、数据布局、缓冲槽位和保存或重算的选择。

评估模型应接收算子约定、输入形状、候选切分和设备信息，给出能够指导实现的判断。它不必从一开始就精确预测每个周期，但应清楚说明假设、主要代价和适用范围。

| 评估维度 | 需要解释什么 |
| --- | --- |
| 语义与依赖 | 用 DAG 标明各阶段、kernel 边界、chunk 内并行和跨 chunk 状态传递。 |
| 计算 | Cube/Vector 工作量，包含归约、指数、类型转换和重计算。 |
| 搬运与缓存 | 逻辑 GM 流量、L2 服务的流量、实际 HBM 读写，以及复用是否成立。 |
| 资源 | L1/L0/UB 同时存活的峰值、缓冲槽位、workspace 与跨核 L2 工作集。 |
| 调度 | 核心分工、同步等待、流水线重叠和负载均衡。 |
| 精度 | 中间 dtype、有损转换位置和端到端误差预算。 |
| 泛化 | 序列长度、head 数或工作集变化时，何时需要切换策略。 |

### 把 L2 驻留纳入模型

例如，一个 `[1,1024,32,128]` 的 fp32 中间张量，写入并完整读回一次对应 16 MiB 写入加 16 MiB 读取的逻辑 GM 流量。但消费者若在数据被淘汰前读取，就可能利用 L2 带宽中转。L2 带宽可以比 HBM 高数倍，具体收益取决于目标设备与访问模式，可参考昇腾官方的[缓存策略说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00014.html)。

需要统计所有活跃核心、缓冲槽位和同期输入输出形成的工作集，并考虑缓存策略、竞争、访问顺序、淘汰和回写。单个 workspace 小于 L2 容量不足以保证命中。评估时分别估计 L2 与 HBM 的服务需求，再考虑依赖和重叠；不要把所有 GM 访问都按 HBM 带宽计算，也不要将两层耗时无条件相加。

### 统一计时口径

**本题只比较完成一次约定计算所需的各次设备 kernel 实际执行耗时之和。** 不计 kernel 启动开销、主机调度、调用间隙或主机内存分配。必要的归一化、布局转换和重计算 kernel 都应计入；kernel 内的数据搬运、同步等待以及流水线启动与排空属于实际执行耗时。

请保留方案真实的数据依赖与执行顺序，记录各 kernel 的执行次数、耗时、预热与重复次数、统计方式，以及缓存冷热状态。跨 kernel 的 L2 复用应在测量中保留；单独反复运行某个 kernel 得到的热缓存结果，应另行标注。

检查器不以 Python 墙钟时间打分，JSON 中 `kernel_timing` 为 `null`，设备执行也需另行举证。请区分模型估计、EasyASC simulator 和 A3 真机数据，在同一 A3 环境与输入集下比较硬件性能。

### 用实验检验预测

至少比较两种有实质差异的候选策略，并保留测量前的预测。优先测量两者；若某个方案在完整实现前被排除，应提供资源计算或聚焦实验解释原因。分析预测与结果的差异，以及模型需要如何修正。

可以从[评估报告模板](kda_a3_exercise/EVALUATION.md)开始，再将其中可重复的工作做成 skill 或工具。例如：估计缓冲区生命周期与 L2 工作集、从 trace 定位依赖瓶颈、生成候选切分、封装 A3 数据桥接，或自动整理实验结果。请让其他开发者能够把方法用于新形状或其他算子。

## 6. 需要交付什么？

**交付物一：切分策略评估模型。** 提交 skill、文档、代码或工具，说明输入输出、设备假设、至少两种候选策略、选择依据、预测与实验对照，以及适用范围和复用方式。

**交付物二：可运行的 A3 chunk prefill 实现。** 提交 EasyASC kernel 源码、完整 `forward` 组合入口、运行环境说明、运行命令、正确性报告，以及解释数据流和 pipe 并行的 trace。性能报告遵循第五节口径，并注明数据来源；硬件结果需附具体 A3 SoC、CANN 版本和测量方法。若实现依赖 EasyASC 修改，同时提供可应用的修改与相应验证。

提交目录由你自行组织，但需要让其他人能够按说明复现，使用同一题包版本、输入和误差规则完成检查。正确性是基础，同时关注策略推理、kernel 实际总耗时、可复现性，以及方法对其他开发者的帮助。

可选扩展包括 `extended` 用例、反向与训练链路、额外 skill 或库能力。题包的 `backward` 是 PyTorch autograd 参考工具，当前提交接口只验前向；扩展结果需单独说明接口、缓存或重计算策略及验证方法。

整理或分发本题资料时，可以从根目录生成同时包含题面、题包、测试与上游许可证的压缩包：

```bash
python build_package.py --output dist/kda_a3_practice_v0.3.0.zip
```

## 扩展 EasyASC 与问题反馈

EasyASC 是 Ascend C 的轻量级 Python 封装，你可以根据需求自行扩展 API，对接 Ascend C 的更多能力。由已有操作组合而成的功能，可以封装为可复用的组合 API；接入新的底层操作时，通常需要在 `easyasc/stub_functions/` 中定义接口、参数校验和指令生成，在 `easyasc/targets/ascendc/` 中实现并注册对应的代码生成逻辑，并在 `easyasc/simulator/` 中补充模拟执行逻辑及管道路由。同时需要接好目标设备的公共 API 导出和同步所需的读写信息；涉及性能估计时，再补充相应的时序模型。

这些路径均相对于 EasyASC 仓库根目录。具体扩展方法可参考 `doc/11_architecture_for_contributors.md` 和 `agent/references/code-paths.md`，并结合相近 API 的实现，验证参数约束、生成代码与 simulator 行为。欢迎将做题过程中新增的 API、组合库和开发工具整理为可复用的贡献。

如果对本习题或 EasyASC 有任何问题，包括接口使用、框架行为、功能需求和题目建议，都欢迎到[本习题仓提交 issue](https://github.com/ddddwee1/kda_a3_practice/issues)。涉及具体错误时，请尽量附上最小复现代码、输入形状、运行环境和相关日志，便于共同定位与讨论。
