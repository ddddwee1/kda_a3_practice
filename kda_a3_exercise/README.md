# A3 KDA Chunk Prefill 社区习题包

版本：`0.3.0`。本题包提供公开模型来源的 PyTorch golden、可执行接口约束、固定测试用例、辅助矩阵和带基础 host 反作弊检查的提交检查器。目标是完成 A3 KDA 的 **chunk prefill**，并交付可复用的切分策略评估方法。重点是多 token 输入的 chunk 内并行计算与跨 chunk 状态传递。

题包提供参考与评测工具；A3 kernel 留给参与者实现。使用本版本的固定用例和误差规则验证实现，并在报告中记录版本。基础交付是 chunk prefill 前向，反向属于可选扩展。

交付物为 A3 kernel 实现和切分策略评估模型。后者可使用 [EVALUATION.md](EVALUATION.md) 模板，也可采用 skill、工具或代码形式。

## 1. Golden 来自哪个公开模型？

来源链如下：

```text
moonshotai/Kimi-Linear-48B-A3B-Instruct
  modeling_kimi.py::KimiDeltaAttention.forward
    -> FLA chunk_kda（prefill 路径）
       use_qk_l2norm_in_kernel=True
    -> FLA naive_chunk_kda 的纯 PyTorch 参考定义
       naive_recurrent_kda 仅用于独立校验
```

公开模型的 [KimiDeltaAttention](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py)在进入 KDA 前生成门控和 beta，在调用中启用 Q/K 的 L2 归一化。其[配置文件](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json)给出 32 个 KDA head、每个 head 128 维。

题包直接收录固定版本的 [FLA PyTorch 参考源码](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py)，保留其 MIT 许可证和作者署名。收录文件的适配仅涉及 Python 3.8 类型标注及元组语法，递推和分块计算体保持原样。[golden.py](golden.py)完成 Q/K 归一化后，按 chunk 调用上游 `naive_chunk_kda`，默认 chunk 大小为 64。主 golden 不会回退到逐 token recurrent。

本题的边界是 **KDA 核心及其 Q/K 归一化**。Q/K/V 投影、短卷积、原始门控激活、输出门控归一化和输出投影均在边界之外；因此不用下载整个模型权重。详情与来源哈希见 [PROVENANCE.md](PROVENANCE.md) 和 [SOURCES.json](upstream/SOURCES.json)。题包不导入 EasyASC，也不使用仓内 KDA 项目的 golden 或测试结果作为标准答案。

### Chunk prefill 的计算主干

一个 chunk 内先计算门控前缀和、带因果掩码的 QK/KK 矩阵，以及三角求解和 WY 中间量。随后用矩阵运算将进入该 chunk 的状态 `S_in` 合入输出，并生成 `S_out` 供下一个 chunk 使用：

```text
chunk 内：gate cumsum -> 因果矩阵与三角求解 -> W、U、Qg、Kg、Aqk
跨 chunk：V_new = U - W @ S_in
          O     = Qg @ S_in + Aqk @ V_new
          S_out = exp(G_last)[:, None] * S_in + Kg.T @ V_new
```

这里 `G` 为 chunk 内的门控累积和，`Qg = scale * q * exp(G)`，`Kg = k * exp(G_last - G)`。具体矩阵构造见上游 `naive_chunk_kda`，参与者可以重新设计切分、融合和调度方式。

主 golden 逐 chunk 调用该公式以限制 CPU 内存占用，chunk 内仍然执行上游的分块矩阵算法。最后不足 64 个 token 时，golden 内部使用零 Q/K/V、零 log gate、零 beta 补齐，使补齐部分不改变状态，再裁掉补齐输出。检查器传给参与者的仍是原始长度，参与者自行在实现中处理尾块。`chunk_prefill(..., chunk_size=32)` 可用于参考对照；提交实现的内部 chunk 大小不受该参考参数限制。

`recurrent_reference` 保留为独立数学校验，可用于检查 chunk 结果和最终状态。单 token decode 循环不是本题的目标实现。由于两种计算顺序的浮点舍入可能不同，对照采用数值误差界，而不要求逐位相等。

## 2. 先跑起来

在已有 `torch210npu` 环境中可以直接运行。新建环境时安装 `requirements.txt` 中的依赖。题包自检仅依赖 PyTorch、einops 和 pytest，不需要 FLA 安装包、Triton、CUDA、CANN 或昇腾硬件。

以下命令在本 README 所在目录执行；压缩包解压后同样适用：

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python check.py --list
python check.py --check-golden --suite core --report results/golden_check.json
```

`--check-golden` 只检查测试流程是否能跑通，报告中明确标为 `golden_harness_check`。数学正确性的独立检查由 pytest 中的 recurrent 对照、单步公式、尾块与状态衔接、梯度测试承担；还检查主 golden 确实调用 chunk 实现。两者都不代表 A3 实现已经通过。

首次编写实现时，可复制 [submission.py](submission.py) 为自己的入口文件，例如 `my_solution.py`。在该文件中实现 `forward`，然后运行：

```bash
python check.py --submission my_solution.py --suite smoke
python check.py --submission my_solution.py --suite core --report results/correctness.json
python check.py --submission my_solution.py --suite extended
```

未实现的模板会明确失败，不会自动调用 golden，也不会跳过测试。退出码为 `0` 表示所选用例全部通过，`1` 表示有用例失败，参数或加载错误返回 `2`。`--case prefill_tail513` 可用于定位单个用例；完整基础题验收需要整个 `core` 集通过。

参赛者自行决定实现目录、执行方式以及 EasyASC kernel 的组合。检查器传入 CPU 张量，适配入口可以组织 `OpExec` 调用及设备传输。开发 EasyASC 实现时，先按 EasyASC 的 quickstart 配置环境，再将其仓库加入 `PYTHONPATH`，例如：

```bash
export EASYASC_ROOT="$HOME/projects/easyasc"
export PYTHONPATH="$EASYASC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

请按实际位置修改路径。适配入口负责组织 EasyASC simulator 或真机执行，检查器不会替参与者实现设备运行链路。需要核对生成的 A3 设备代码时，可使用 `debug=True` 调试工程的 CAModel `sim` 模式，不能使用 `cannsim record` 路径。CAModel 仿真耗时很长，仅用于小规模用例的正确性验证，无需在报告中提交或标注其结果，也不用于性能评估。

## 3. 可执行的接口约定

[contract.json](contract.json)固定版本、dtype、归一化和误差规则；[contract.py](contract.py)执行结构及数值检查。

```python
def forward(q, k, v, g, beta, initial_state=None, *, aux=None):
    # Implement the kernel composition and return these two tensors.
    return o, final_state
```

| 参数 | 形状 | dtype 与语义 |
| --- | --- | --- |
| `q`, `k` | `[B,T,H,K]` | bf16；未经 L2 归一化 |
| `v` | `[B,T,HV,V]` | bf16 |
| `g` | `[B,T,HV,K]` | fp32；已激活的自然对数衰减增量，`g <= 0`，未经 cumsum |
| `beta` | `[B,T,HV]` | fp32；已激活的更新权重，范围 `[0,1]` |
| `initial_state` | `[B,HV,K,V]` | fp32；`None` 表示零状态 |
| `o` | `[B,T,HV,V]` | bf16 |
| `final_state` | `[B,HV,K,V]` | fp32；始终返回 |

输入和输出均为连续布局，维度均为正整数，`HV % H == 0`。固定 query 缩放为 `K**-0.5`。当 `HV > H` 时，每个 Q/K head 对应连续的一组 value head，映射为 `value_head // (HV // H)`。

Q/K 归一化必须由提交实现承担。其 PyTorch 语义是：

```text
x_float = x.float()
inverse_norm = 1.0 / sqrt(sum(x_float * x_float, dim=-1, keepdim=True) + 1e-6)
x_norm = (x_float * inverse_norm).bfloat16()
```

归一化结果先存为 bf16，再进入上游 fp32 chunk KDA 计算。epsilon 加在平方和内部。`norm_epsilon` 和零行用例用于区分省略 epsilon、错误放置 epsilon 或重复归一化等实现差异。具体算法以 [golden.py](golden.py) 和[上游分块源码](upstream/fla_naive.py)为准。

测试脚本生成输入后，会给 golden 和提交实现分别传入相同值、相同布局的副本，并检查输入是否被修改。检查器不会预先归一化、展开 head、重排或缩放待测输入。实现如需这些计算，必须自行完成；参与者应按题目要求让数值工作运行在提交的 kernel 内。

### 评测方提供的辅助矩阵

`aux` 是评测脚本额外传入的只读字典，由 [auxiliary.py](auxiliary.py)预先生成。它只依赖公开尺寸，不依赖 Q/K/V/g/beta 或 golden 结果。每种矩阵都提供 **32、64、128 阶**版本，dtype 为 **bf16**，布局连续：

| 访问方式 | 定义 | 对角线 |
| --- | --- | --- |
| `aux[L]["lower"]` | `j <= i` 时为 1，否则为 0 | 包含 |
| `aux[L]["upper"]` | `i <= j` 时为 1，否则为 0 | 包含 |
| `aux[L]["ones"]` | 所有元素均为 1 | 包含 |
| `aux[L]["identity"]` | 单位矩阵 | 为 1，其余为 0 |

例如，对形状为 `[L,K]` 的数据 `X`，沿 token 维的包含当前位置的前缀和可写为 `lower @ X`；若采用 `[K,L]` 布局，则可写为 `X @ upper`。这些乘法需要在提交的 Cube kernel 中执行，host 侧只需从字典中取得对应矩阵，传入 kernel。全 1 矩阵可以用于归约和广播类构造，单位矩阵可辅助三角系统处理。

矩阵使用与否由参与者决定，golden 不依赖这些辅助输入。评测方的常量构造不计时，但 kernel 加载、使用或转换矩阵的实际耗时仍计入。每个用例收到独立副本，脚本检查调用后 Python 张量的值、dtype、形状和字典结构未被修改。设备端也应按只读输入使用，写回结果需另分配工作区。

### Host 操作白名单

[host_guard.py](host_guard.py)在提交模块导入和 `forward` 执行期间检查操作。基础策略如下：

| 允许 | 不允许 |
| --- | --- |
| `view`、无需拷贝的 `reshape`、`flatten`、`squeeze`、`unsqueeze`、`detach` 等重解释 | 加减乘除、matmul、cumsum、归约、归一化、exp 等数值计算 |
| `torch.empty`、`empty_like`、`empty_strided`、`new_empty` 等未初始化分配 | `zeros`、`ones`、`full`、`arange`、随机数或从列表创建数值张量 |
| 查询 shape/dtype/device 等元信息，使用 Python 计算分块尺寸 | `.item()`、`.tolist()`、`.numpy()`、storage/data pointer 等数据提取 |
| 必要的同 dtype CPU/NPU 传输 | dtype 转换、clone、需要拷贝的 reshape、切片/索引、transpose/permute、cat/gather 等数据或布局处理 |
| 获取评测方提供的 `aux` 矩阵并传给 kernel | 在 host 侧自行构造辅助矩阵，或使用辅助矩阵做 host 数值计算 |

上述限制作用于参赛代码。正常 `OpExec` 内部的输入输出搬运和 simulator 计算单独识别；仅因某段参赛回调位于 `OpExec` 调用栈内，并不会得到豁免。检查还要求每个提交用例观察到受信任的 `OpExec` 运行时操作。其他设备执行入口需要组织方审查并扩展支持，不能直接关闭检查。

提交模块及其本地依赖会做基础源码检查，拒绝导入题包内部参考、NumPy、ctypes、线程/进程计算入口和 dispatch 禁用工具等常见绕过路径。建议将独立自测代码与提交模块分开。违规操作即使被提交代码捕获异常，仍会导致该用例失败。JSON 报告记录导入审计、允许的 host 操作、运行时操作和违规项目；`--submission` 没有关闭审计的命令行选项。

这是同进程的基础检查，不是针对任意恶意 Python 的安全沙箱。组织方需要固定评测脚本与 EasyASC 框架副本，并结合源码审查、设备执行证据和必要的隔离评测确认提交行为。纯 golden 自检明确标为 `reference_self_check`，不适用参赛代码的 host 限制，也不能用作提交通过证明。

## 4. 固定用例与误差规则

[cases.json](cases.json)是用例清单，[cases.py](cases.py)按每个用例的随机种子生成输入。Q/K 是带有不同 token 幅度的未归一化随机向量，V 使用单位量级随机值，非零初态标准差为 `0.25`。门控采用上游测试使用的 `logsigmoid` 生成方式，并覆盖不同衰减强度；beta 覆盖随机值及边界值。

这些是合成算子输入，不是从模型权重运行中采集的激活。组织者固定生成逻辑、种子和 PyTorch 版本后即可复现。检查报告记录 PyTorch 版本，以及约束文件、用例清单和提交文件的 SHA-256。

| 用例集 | 数量 | 用途 |
| --- | ---: | --- |
| `smoke` | 3 | 128、512、513 token 的多 chunk prefill；快速检查接线，属于 core 的子集。 |
| `core` | 16 | 基础题：`H=HV`、`K=V=128`；长度 128–2048，覆盖多 chunk、尾块、多 batch/head、初态、门控、beta 和归一化边界。 |
| `extended` | 3 | 可选扩展：255–259 token 的分组 value head、不等 K/V、非整齐维度。 |
| `performance` | 2 | 模型配置形状：32 head、128 维，序列长度 1024 和 4096；该名称表示用途，检查器仍只验正确性。 |

用例不规定内部 chunk 大小、kernel 数量或中间布局。基础题要求正确处理尾块，并以 chunk prefill 作为实现与评估重点。单 token 和短序列只存在于 golden 的边界自检中，不作为主评测负载。可通过 `python check.py --suite performance --list` 查看模型形状。

每个输出同时满足以下两种误差条件才算通过：

1. 逐元素满足 `abs(actual - golden) <= atol + rtol * abs(golden)`。
2. 当 golden 非全零时，满足 `||actual - golden||₂ / ||golden||₂ <= relative_l2`；全零参考仅使用绝对误差条件。

| 输出 | atol | rtol | relative_l2 |
| --- | ---: | ---: | ---: |
| `o` | 0.02 | 0.02 | 0.02 |
| `final_state` | 0.02 | 0.02 | 0.02 |

第二个条件可拒绝“因输出幅度小，全零实现也满足绝对误差”的情况。检查器同时拒绝 NaN/Inf、错误形状、错误 dtype、不连续输出、遗漏状态或修改输入。失败报告提供各张量的最大绝对误差、相对 L2 误差和超过逐元素阈值的数量。

`atol=rtol=2e-2` 参考 EasyASC 现有 A5 KDA 前向检查脚本中的统一阈值，来源记录在 `contract.json` 的 `tolerances_basis` 中；额外的相对 L2 上限为 **2%**。绝对误差地板用于减少接近零的输出对实现精度的过度约束。Golden 来自公开模型的 FLA 定义。

题目不要求中间缓冲、状态存储或全部计算使用 fp32，参与者可自行选择 bf16/fp16 等计算路径，只需满足公开 I/O dtype 和准出误差。自检包含长序列中使用 bf16 保存跨 chunk 状态的参考变体。请使用本版本的统一阈值进行比较。

## 5. 反向与性能的边界

[golden.py](golden.py)额外提供 `backward`，通过 PyTorch autograd 对本题前向求导，损失为：

```text
sum(o.float() * do.float()) + sum(final_state * dht)
```

`do` 与 `o` 同形同 dtype，`dht` 与 `final_state` 同形同 dtype；返回 `dq/dk/dv/dg/dbeta/dh0`，各梯度与对应输入 dtype 一致，初态不存在时 `dh0=None`。它包含 Q/K 归一化的梯度，是本题 PyTorch 前向的导数，不承诺与优化后的 FLA backward 逐位一致。

反向是可选参考工具，不纳入本版本的提交验收接口。开展训练扩展时，请另行说明梯度接口、缓存或重计算策略和精度验证方法。

检查器不使用 Python 墙钟时间评估性能。正式性能报告只统计设备上各 kernel 的实际执行耗时及其总和，包括 Q/K 归一化、必要的转换和重计算 kernel；不计 kernel 启动、主机调度或内存分配。测量时保留实际执行顺序和 L2 复用条件。

结果 JSON 中的 `kernel_timing` 为 `null`，`device_execution_verified` 为 `false`：基础 host 审计不能单独证明代码实际运行在 A3，也不能区分等价的 recurrent 与 chunk 算法。A3 实现是否组织了 chunk 内并行，以及 trace 与设备 profiling 证据，仍需单独评审。

## 6. 文件与分发

| 文件 | 作用 |
| --- | --- |
| [golden.py](golden.py) | Chunk prefill 主 golden、recurrent 独立校验和可选反向工具。 |
| [upstream/fla_naive.py](upstream/fla_naive.py) | 固定版本的上游递推与分块参考。 |
| [upstream/LICENSE.fla](upstream/LICENSE.fla)、[PROVENANCE.md](PROVENANCE.md) | 许可证、来源链及适配说明。 |
| [contract.json](contract.json)、[contract.py](contract.py) | 可执行约束与比较器。 |
| [cases.json](cases.json)、[cases.py](cases.py) | 固定用例及输入生成。 |
| [auxiliary.py](auxiliary.py) | 下三角、上三角、全 1、单位矩阵及只读检查。 |
| [host_guard.py](host_guard.py) | Host 白名单、数据提取拦截和基础源码审计。 |
| [submission.py](submission.py)、[check.py](check.py) | 实现模板与检查器。 |
| [EVALUATION.md](EVALUATION.md) | 切分策略评估与实验报告模板。 |
| [tests/](tests/) | golden 的独立验证和检查器回归测试。 |
| [build_package.py](build_package.py) | 生成包含源码、测试及许可证的确定性 ZIP 和 SHA-256。 |

从本 README 所在目录重新打包独立题包：

```bash
python build_package.py --output dist/kda_a3_chunk_prefill_v0.3.0.zip
```

解压得到 `kda_a3_exercise/`。仅运行 golden 与自检时，这个目录可脱离原仓库使用；开发参赛 EasyASC kernel 时，再将 EasyASC 仓库放到 `PYTHONPATH` 中。

完整习题目录还包含根目录的总体说明和打包脚本；在完整目录根部运行 `python build_package.py --output dist/kda_a3_practice_v0.3.0.zip`，可将题面与题包一同分发。

初次验证环境为 Python 3.11.15、PyTorch 2.7.1、einops 0.8.2、pytest 9.0.2（CPU）。源码另外检查了 Python 3.8 语法兼容性，未在 Python 3.8 解释器上执行。
