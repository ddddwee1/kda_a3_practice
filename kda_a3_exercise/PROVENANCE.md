# Golden 来源与适配记录

核验日期：2026-09-12。原始文件已经从公开地址实际下载并检查；各文件 URL、固定版本及 SHA-256 记录在 [upstream/SOURCES.json](upstream/SOURCES.json)。题包运行时不联网，也不导入仓内 KDA 项目的任何 reference。

## 公开模型调用链

- 模型：[moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)。
- 固定版本：`e1df551a447157d4658b573f9a695d57658590e9`。
- [modeling_kimi.py](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py)第 25 行导入 FLA KDA，`KimiDeltaAttention.forward` 在第 560–561 行生成激活后的门控和 beta，第 568–591 行调用 KDA 并启用 Q/K L2 归一化。
- [config.json](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json)中的 `linear_attn_config` 定义 32 个 head、128 维。题包据此提供模型形状用例；较小形状用于降低开发验证成本。

模型的短卷积、线性投影、门控激活和输出层没有被复制进题包。输入边界选择在模型调用 KDA 时，因此必须执行 Q/K 归一化，但接收已经激活的 `g` 与 `beta`。本题聚焦模型的 `chunk_kda` prefill 路径，默认测试长度均大于模型切换到 `fused_recurrent` 的短序列区间。测试张量是合成数据，未读取模型权重。

## PyTorch 定义

FLA 固定版本：`516143e31fce09925e6c39ac37148444bad176c4`。这与模型源码版本分别固定，用于题目复现，并不声称它就是该模型训练时使用的 FLA 版本。

| 上游文件 | 采用的内容 |
| --- | --- |
| [fla/ops/kda/naive.py](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py) | 直接收录 `naive_chunk_kda` 与 `naive_recurrent_kda`；chunk 作为主 golden 的计算核心，recurrent 仅用于独立验证。 |
| [fla/ops/kda/chunk.py](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/chunk.py) | 核对归一化发生在递推之前、默认缩放、输入形状及最终状态语义。 |
| [fla/modules/l2norm.py](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/modules/l2norm.py) | 核对 `1/sqrt(sum(x*x)+1e-6)`、fp32 计算和同 dtype 写回，并在 golden 中用 PyTorch 表达。 |
| [tests/ops/test_kda.py](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/tests/ops/test_kda.py) | 借鉴随机输入、`logsigmoid` 门控、状态与输出共同参与梯度损失的测试方法；没有引入其 Triton/GPU 测试依赖。 |

收录的 [fla_naive.py](upstream/fla_naive.py)只做了两项语法适配：以 `typing.Optional` 替换 `T | None`，为包含星号展开的元组表达式添加括号。所有函数计算体保留上游语义。原始与适配后的源文件哈希均有记录，原始 MIT 许可证保存为 [LICENSE.fla](upstream/LICENSE.fla)，原作者版权声明保留在源码中。

归一化包装、分块调用、尾块补齐、接口校验、合成用例和提交检查器是本题包的适配层。主 golden 按默认 64 token 的 chunk 调用上游分块公式，并在 chunk 之间传递状态，以限制 CPU 内存占用；不会回退到逐 token recurrent。尾块以零 Q/K/V、零 log gate、零 beta 补齐，裁剪输出并保留最终状态。该包装不改变检查器传给参与者的输入长度。

Golden 在 CPU 上采用 PyTorch 的浮点运算。自检用上游 recurrent、32/64 两种分块和不同尾块边界交叉验证输出与状态，不要求不同归约顺序逐位一致。不宣称已经验证 Triton/CUDA、A3 CAModel 或真机逐位一致。
