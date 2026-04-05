"""
Transformer 解码器层

整合所有组件：
1. 掩码多头自注意力
2. 残差连接 + 层归一化
3. 交叉注意力到编码器输出
4. 残差连接 + 层归一化
5. 位置前馈网络
6. 残差连接 + 层归一化

架构：
    输入
     ↓
    [掩码自注意力] → Add & Norm
     ↓
    [交叉注意力到编码器] → Add & Norm
     ↓
    [前馈网络] → Add & Norm
     ↓
    输出
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层

    【这是什么？】
    这是 Transformer 解码器的核心构建块！
    完整的解码器由多个 DecoderLayer 堆叠而成（通常是 6 层）

    【与编码器的关键区别】
    编码器有 2 个子层，解码器有 3 个子层：

    编码器：
        1. 自注意力（看整个输入句子）
        2. 前馈网络

    解码器：
        1. 掩码自注意力（只能看已生成的部分，不能看未来）
        2. 交叉注意力（看编码器的输出，获取源语言信息）
        3. 前馈网络

    【完整架构】
        输入 x (batch, tgt_len, d_model)
         ↓
        ┌─────────────────────────────────────┐
        │  掩码多头自注意力                    │  ← 子层 1: 看已生成的词
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + MaskedAttention(x), 然后归一化
         ↓
        ┌─────────────────────────────────────┐
        │  多头交叉注意力                      │  ← 子层 2: 从编码器获取信息
        │  (Q 来自解码器, K,V 来自编码器)      │
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + CrossAttention(x, enc), 然后归一化
         ↓
        ┌─────────────────────────────────────┐
        │  前馈网络 (FFN)                      │  ← 子层 3: 处理信息
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x), 然后归一化
         ↓
        输出 (batch, tgt_len, d_model)

    【什么是掩码自注意力？】
    问题：在语言生成中，我们一次生成一个词
    - 生成第 3 个词时，我们只看到了第 0, 1, 2 个词
    - 我们不能看到第 4, 5, 6... 个词（它们还不存在！）

    解决方案：使用因果掩码（也叫前瞻掩码）
    - 处理位置 i 时，只能注意到位置 ≤ i 的内容
    - 防止"作弊"看到未来的词

    具体例子："我 喜欢 吃 苹果"
        位置 0 ("我"):     能看到: ["我"]
        位置 1 ("喜欢"):   能看到: ["我", "喜欢"]
        位置 2 ("吃"):     能看到: ["我", "喜欢", "吃"]
        位置 3 ("苹果"):   能看到: ["我", "喜欢", "吃", "苹果"]

    掩码矩阵 (1 = 能看到, 0 = 看不到):
        [[1, 0, 0, 0],  ← 位置 0 只能看到位置 0
         [1, 1, 0, 0],  ← 位置 1 能看到位置 0-1
         [1, 1, 1, 0],  ← 位置 2 能看到位置 0-2
         [1, 1, 1, 1]]  ← 位置 3 能看到位置 0-3

    【什么是交叉注意力？】
    目的：让解码器"看到"编码器的输出
    - 在翻译中：解码器看源句子（英语）
                来生成目标句子（法语）

    机制：与自注意力不同！
    - 自注意力：Q, K, V 都来自同一个输入（解码器）
    - 交叉注意力：Q 来自解码器，K 和 V 来自编码器

    类比：
        自注意力：  "我到目前为止说了什么？"
        交叉注意力："原始句子说了什么？"

    具体例子（英语→法语翻译）：
        编码器输入：  "I love eating apples"
        解码器生成中："J'aime manger ..."

        生成 "manger" (eating) 时：
        - 掩码自注意力：看 ["J'aime"]
        - 交叉注意力：  看 ["I", "love", "eating", "apples"]
                       发现 "eating" 最相关！
        → 帮助生成正确的词 "manger"

    【为什么是这个顺序？】
    1. 先掩码自注意力：理解我们已经生成的内容
       - 看目标序列到目前为止生成的内容
       - 从之前生成的词中构建上下文

    2. 然后交叉注意力：从源获取信息
       - 看编码器输出（源句子）
       - 找到现在哪些源词相关

    3. 最后前馈网络：处理组合信息
       - 组合"我们已生成的"和"源句子"的信息
       - 非线性变换
       - 为下一层或最终预测做准备

    【完整流程示例】
    假设：英语→法语翻译
    源：  "I love apples" (已经被编码器编码)
    目标到目前为止："J'aime" (法语 "I love")
    现在生成：下一个词（应该是 "les" 之类的）

    输入：
        x = ["J'", "aime", "<CURRENT>"] 的嵌入
        x.shape = (1, 3, 512)  # batch=1, tgt_len=3, d_model=512
        encoder_output = 编码后的 "I love apples"
        encoder_output.shape = (1, 3, 512)  # batch=1, src_len=3, d_model=512

    步骤 1：掩码自注意力
        - "J'" 看：["J'"]
        - "aime" 看：["J'", "aime"]
        - "<CURRENT>" 看：["J'", "aime", "<CURRENT>"]
        → 每个词知道它之前的内容
        → masked_attn_output.shape = (1, 3, 512)

    步骤 2：Add & Norm（第一次）
        - x = x + masked_attn_output （残差）
        - x = LayerNorm(x) （归一化）
        → x.shape = (1, 3, 512)

    步骤 3：交叉注意力
        - Q 来自解码器："我在源中寻找什么？"
        - K, V 来自编码器："这是源句子的内容"
        - "<CURRENT>" 可能高度注意 "apples"
        → cross_attn_output.shape = (1, 3, 512)

    步骤 4：Add & Norm（第二次）
        - x = x + cross_attn_output （残差）
        - x = LayerNorm(x) （归一化）
        → x.shape = (1, 3, 512)

    步骤 5：前馈网络
        - 独立处理每个位置
        - 512 → 2048 → 512 （扩展→变换→压缩）
        → ff_output.shape = (1, 3, 512)

    步骤 6：Add & Norm（第三次）
        - x = x + ff_output （残差）
        - x = LayerNorm(x) （归一化）
        → x.shape = (1, 3, 512)

    最终输出：(1, 3, 512) ← 与输入形状相同

    参数：
        d_model: 模型维度（例如 512）
        num_heads: 注意力头数（例如 8）
        d_ff: 前馈网络隐藏维度（例如 2048，通常是 d_model 的 4 倍）
        dropout: Dropout 比率（默认 0.1）
        activation: 前馈网络激活函数（'relu' 或 'gelu'）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # ========== 组件 1：掩码多头自注意力 ==========
        # 这是第一个子层，负责"看已生成的内容"
        #
        # 自注意力意味着：
        # - Query, Key, Value 都来自解码器输入（自我注意）
        # - 每个已生成的词可以看到之前生成的词
        # - 不能看到未来的词（掩码）
        #
        # 为什么要掩码？
        # - 在推理时，未来的词还不存在！
        # - 在训练时，我们掩码来模拟这个条件
        # - 这是自回归生成（一次一个词）
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 组件 2：多头交叉注意力 ==========
        # 这是第二个子层，负责"看源语言信息"
        #
        # 交叉注意力意味着：
        # - Query 来自解码器（我在寻找什么？）
        # - Key 和 Value 来自编码器（这是源信息）
        # - 与自注意力不同，Q, K, V 不全来自同一个源
        #
        # 为什么需要？
        # - 解码器需要知道源句子说了什么！
        # - 例如：翻译 "eating" → 需要回头看英语的 "eating"
        # - 这是解码器"注意到"输入序列的方式
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 组件 3：位置前馈网络 ==========
        # 这是第三个子层，负责"处理信息"
        #
        # 与编码器的前馈网络相同：
        # - 对每个位置进行独立的非线性变换
        # - 提取更复杂的特征
        # - 增加模型表达能力
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== 组件 4, 5, 6：三个层归一化层 ==========
        # 为什么是三个？
        # - 因为我们有三个子层（掩码自注意力、交叉注意力、前馈网络）
        # - 每个子层都需要一个 LayerNorm
        #
        # 目的：与编码器相同
        # - 归一化：将每个样本标准化为 mean=0, std=1
        # - 稳定训练：防止数值爆炸或消失
        # - 加快收敛：使梯度更稳定
        self.norm1 = nn.LayerNorm(d_model)  # 第一个子层（掩码自注意力）
        self.norm2 = nn.LayerNorm(d_model)  # 第二个子层（交叉注意力）
        self.norm3 = nn.LayerNorm(d_model)  # 第三个子层（前馈网络）

        # ========== 组件 7, 8, 9：三个 Dropout 层 ==========
        # 为什么是三个？
        # - 因为我们有三个子层
        # - 每个子层的输出都需要 Dropout（在残差之前）
        #
        # 目的：与编码器相同
        # - 防止过拟合
        # - 使模型不过度依赖某些路径
        self.dropout1 = nn.Dropout(dropout)  # 掩码自注意力
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码器层前向传播

        参数：
            x: 目标序列输入，形状 (batch_size, tgt_len, d_model)
               （通常是目标的词嵌入 + 位置编码）

            encoder_output: 编码器输出，形状 (batch_size, src_len, d_model)
                           （编码后的源序列，例如英语句子）

            tgt_mask: 目标掩码，形状 (batch_size, 1, tgt_len, tgt_len)
                     组合：
                     1. 填充掩码：忽略目标中的 <PAD> 词元
                     2. 因果掩码：防止看到未来的词元
                     用于掩码自注意力

            src_mask: 源填充掩码，形状 (batch_size, 1, 1, src_len)
                     忽略源序列中的 <PAD> 词元
                     用于交叉注意力

        返回：
            output: 解码器层输出，形状 (batch_size, tgt_len, d_model)
                   （与输入维度相同）

        完整流程：
            1. 掩码自注意力：每个目标词元只注意之前的词元
            2. Add & Norm：残差连接 + 层归一化
            3. 交叉注意力：目标注意源（编码器输出）
            4. Add & Norm：残差连接 + 层归一化
            5. 前馈网络：独立处理每个词元，非线性变换
            6. Add & Norm：残差连接 + 层归一化

        具体例子（英语→法语翻译）：
            源（英语）："I love apples"
            目标（法语）："J'aime les pommes"

            encoder_output = 编码后的("I love apples")
            encoder_output.shape = (1, 3, 512)

            训练时，目标输入 = "J'aime les pommes"
            x.shape = (1, 4, 512)  # batch=1, tgt_len=4, d_model=512

            子层 1：掩码自注意力
                - "J'" 看：["J'"] 仅此
                - "aime" 看：["J'", "aime"]
                - "les" 看：["J'", "aime", "les"]
                - "pommes" 看：["J'", "aime", "les", "pommes"]
                → 每个词知道它之前的上下文
                → masked_attn_output.shape = (1, 4, 512)

            Add & Norm 1:
                - x = x + masked_attn_output （残差）
                - x = LayerNorm(x) （归一化）
                → x.shape = (1, 4, 512)

            子层 2：交叉注意力
                - Q 来自解码器：["J'", "aime", "les", "pommes"]
                - K, V 来自编码器：["I", "love", "apples"]
                - "pommes" 高度注意 "apples"
                - "aime" 高度注意 "love"
                → 获取源信息
                → cross_attn_output.shape = (1, 4, 512)

            Add & Norm 2:
                - x = x + cross_attn_output （残差）
                - x = LayerNorm(x) （归一化）
                → x.shape = (1, 4, 512)

            子层 3：前馈网络
                - 独立处理每个词
                - 512 → 2048 → 512 （扩展→变换→压缩）
                → ff_output.shape = (1, 4, 512)

            Add & Norm 3:
                - x = x + ff_output （残差）
                - x = LayerNorm(x) （归一化）
                → x.shape = (1, 4, 512)

            最终输出：(1, 4, 512)
        """
        # ========== 子层 1：掩码多头自注意力 ==========

        # 步骤 1：掩码自注意力
        # Q = K = V = x（三个输入都是解码器输入，因此是"自"注意力）
        # 但使用 tgt_mask 来防止看到未来位置
        #
        # 这做什么？
        # - 每个目标词元注意自己和之前的词元
        # - 不能注意未来的词元（生成时它们不存在！）
        # - 从到目前为止生成的内容构建上下文
        #
        # 具体例子（"J'aime les pommes"）：
        # - "J'" (pos 0) 注意：
        #   * "J'" (pos 0) ✓ （能看到）
        #   * "aime" (pos 1) ✗ （未来，掩码）
        #   * "les" (pos 2) ✗ （未来，掩码）
        #   * "pommes" (pos 3) ✗ （未来，掩码）
        #
        # - "les" (pos 2) 注意：
        #   * "J'" (pos 0) ✓ （过去，能看到）
        #   * "aime" (pos 1) ✓ （过去，能看到）
        #   * "les" (pos 2) ✓ （当前，能看到）
        #   * "pommes" (pos 3) ✗ （未来，掩码）
        #
        # tgt_mask 目的：
        # 1. 因果掩码：防止看到未来（下三角矩阵）
        # 2. 填充掩码：如果目标有填充，忽略 <PAD> 词元
        masked_attn_output = self.self_attention(x, x, x, tgt_mask)
        # masked_attn_output.shape = (batch_size, tgt_len, d_model)

        # 步骤 2：Dropout + 残差连接
        #
        # Dropout:
        # - 训练时：随机将一些值设为 0
        # - 防止过拟合
        masked_attn_output = self.dropout1(masked_attn_output)

        # 残差连接：
        # x = x + MaskedAttention(x)
        #     ↑            ↑
        #  原始输入    掩码注意力输出
        #
        # 为什么要加原始输入 x？
        # 1. 梯度流动：梯度可以直接通过 x 流动（捷径）
        # 2. 更容易学习：模型只需学习"修改"（增量）
        # 3. 保留信息：即使注意力学习不好，x 仍然在那里
        x = x + masked_attn_output  # 第一个残差连接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步骤 3：层归一化
        #
        # 归一化每个样本的每个位置
        # 公式：output = (x - mean) / std * gamma + beta
        #
        # 为什么需要？
        # 1. 稳定数值范围：防止数值爆炸/消失
        # 2. 加快收敛：稳定的输入分布
        # 3. 层独立性：每层可以更独立地学习
        x = self.norm1(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== 子层 2：多头交叉注意力 ==========

        # 步骤 4：交叉注意力
        #
        # 这与自注意力不同！
        # - Query (Q)：来自解码器 (x) - "我在源中寻找什么？"
        # - Key (K)：来自编码器 (encoder_output) - "源中有什么可用？"
        # - Value (V)：来自编码器 (encoder_output) - "实际的源内容"
        #
        # 这做什么？
        # - 解码器问："源句子的哪一部分现在相关？"
        # - 编码器提供：源句子的信息
        # - 注意力机制找到匹配
        #
        # 具体例子（英语→法语）：
        # 源（encoder_output）："I love apples"
        # 目标（x）："J'aime ..."
        #
        # 当解码器处理 "aime"（法语 "love"）时：
        # - Q 来自 "aime"："我是法语的 'love'，我的英语源是什么？"
        # - K 来自编码器：["I", "love", "apples"]
        # - 注意力权重：[0.1, 0.8, 0.1]  ← "aime" 高度注意 "love"！
        # - V 按注意力加权 → 主要获取 "love" 的信息
        #
        # 这就是解码器"知道"翻译哪个源词的方式！
        #
        # 参数：
        # - query=x：来自解码器（目标）
        # - key=encoder_output：来自编码器（源）
        # - value=encoder_output：来自编码器（源）
        # - mask=src_mask：忽略源中的 <PAD>
        cross_attn_output = self.cross_attention(
            query=x,                    # Q 来自解码器
            key=encoder_output,         # K 来自编码器
            value=encoder_output,       # V 来自编码器
            mask=src_mask               # 忽略源填充
        )
        # cross_attn_output.shape = (batch_size, tgt_len, d_model)
        # 注意：输出长度 = query 长度 (tgt_len)，不是 key 长度！

        # 步骤 5：Dropout + 残差连接（第二个残差）
        #
        # 与之前的模式相同：
        # 1. Dropout：防止过拟合
        # 2. 残差：x + CrossAttention(x, encoder)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = x + cross_attn_output  # 第二个残差连接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步骤 6：层归一化（第二次归一化）
        #
        # 再次归一化，原因同上
        x = self.norm2(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== 子层 3：位置前馈网络 ==========

        # 步骤 7：前馈网络
        #
        # 与编码器的前馈网络相同：
        # - 对每个位置进行独立的非线性变换
        # - 与注意力不同（注意力看其他位置），前馈网络只看当前位置
        # - 但所有位置共享相同的前馈网络权重
        #
        # 架构：
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # 为什么需要？
        # - 注意力只"重新排列"信息（加权平均）
        # - 前馈网络提供"非线性变换"
        # - 允许模型学习更复杂的特征
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, tgt_len, d_model)

        # 步骤 8：Dropout + 残差连接（第三个残差）
        #
        # 流程与上面类似：
        # 1. Dropout：防止过拟合
        # 2. 残差：x + FFN(x)
        ff_output = self.dropout3(ff_output)
        x = x + ff_output  # 第三个残差连接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步骤 9：层归一化（第三次归一化）
        #
        # 最终归一化
        x = self.norm3(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # 最终输出：
        # - 形状与输入相同：(batch_size, tgt_len, d_model)
        # - 但内容已被以下方式变换：
        #   * 掩码自注意力（来自之前词的上下文）
        #   * 交叉注意力（来自源的信息）
        #   * 前馈网络（非线性处理）
        # - 可以继续到下一个 DecoderLayer，或输出最终预测
        return x


class Decoder(nn.Module):
    """
    完整的 Transformer 解码器

    【这是什么？】
    这是完整的解码器！
    由多个 DecoderLayer 堆叠而成（原始论文使用 6 层）

    【架构图】
        输入 (batch, tgt_len, d_model)
         ↓
        ┌─────────────────┐
        │  DecoderLayer 1 │  ← 层 1：学习基本模式
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 2 │  ← 层 2：学习中级模式
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 3 │  ← 层 3：学习高级模式
        └─────────────────┘
         ↓
           ... (更多层)
         ↓
        ┌─────────────────┐
        │  DecoderLayer N │  ← 层 N：学习最抽象的模式
        └─────────────────┘
         ↓
        层归一化  ← 最终归一化
         ↓
        输出 (batch, tgt_len, d_model)

    【解码器 vs 编码器：有什么区别？】

    编码器：
    - 目的：理解源句子
    - 输入：源词元（例如英语）
    - 注意力：双向的（可以看到整个句子）
    - 输出：源的丰富表示
    - 用于：编码输入用于翻译、分类等

    解码器：
    - 目的：生成目标句子
    - 输入：目标词元（例如法语）
    - 注意力：单向的（只能看到之前的词元）
    - 输出：用于下一个词元预测的表示
    - 用于：自回归生成（一次一个词元）

    【为什么堆叠多层？】
    与编码器的推理相同：

    层 1：局部模式
         - 简单的词关系
         - 基本语法

    层 2-3：中级模式
           - 短语结构
           - 语义角色

    层 4-6：高级语义
           - 句子级意义
           - 源和目标之间的复杂依赖关系

    每一层都建立在前一层之上，学习越来越抽象的特征。

    【解码器如何使用编码器输出】
    每个 DecoderLayer 通过交叉注意力接收 encoder_output：
    - 编码器运行一次：编码整个源句子
    - 解码器运行多次：生成时一次一个词元
    - 每个解码器层都查看编码器输出以获取源信息

    可以这样理解：
    - 编码器："这是英语句子的意思"（运行一次）
    - 解码器："让我在生成法语时检查英语"
              （在每个生成步骤都检查编码器输出）

    【训练 vs 推理】

    训练（教师强制）：
    - 输入：整个目标句子 "J'aime les pommes"
    - 使用因果掩码来模拟生成
    - 所有位置并行处理（高效！）
    - 所有位置同时计算损失

    推理（自回归生成）：
    - 开始："<START>"
    - 生成："J'" → "J' aime" → "J' aime les" → "J' aime les pommes"
    - 一次一个词元（较慢）
    - 使用前一个输出作为下一个输入

    参数：
        num_layers: 解码器层数（原始论文使用 6）
        d_model: 模型维度（例如 512）
        num_heads: 注意力头数（例如 8）
        d_ff: 前馈网络隐藏维度（例如 2048）
        dropout: Dropout 比率（默认 0.1）
        activation: 前馈网络激活函数（'relu' 或 'gelu'）
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # ========== 创建多个 DecoderLayer ==========
        # 使用 nn.ModuleList 存储多个层
        #
        # 为什么用 nn.ModuleList？
        # - 自动注册所有子模块（层）的参数
        # - 让 PyTorch 知道这些层是模型的一部分
        # - 这样优化器可以找到并更新这些参数
        #
        # 为什么不用普通的 Python 列表？
        # - 普通列表：PyTorch 不知道里面有参数
        # - nn.ModuleList：PyTorch 自动注册参数
        #
        # 列表推导式：
        # [DecoderLayer(...) for _ in range(num_layers)]
        # 创建 num_layers 个 DecoderLayer
        # 每个 DecoderLayer 有相同的结构但独立的参数
        #
        # 示例 num_layers=6：
        # self.layers[0] ← 层 1（参数 A）
        # self.layers[1] ← 层 2（参数 B，与 A 不同）
        # self.layers[2] ← 层 3（参数 C，与 A、B 不同）
        # ...
        # self.layers[5] ← 层 6（参数 F）
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # ========== 最终层归一化 ==========
        # 为什么最后还要一个 LayerNorm？
        #
        # 1. 稳定最终输出：
        #    - 经过多层操作后，数值范围可能不稳定
        #    - 最终的 LayerNorm 确保输出分布稳定
        #
        # 2. 更容易进行下游处理：
        #    - 如果连接到输出投影（例如词汇预测）
        #    - 稳定的输入帮助线性层学习得更好
        #
        # 3. 实证性能更好：
        #    - 原始论文使用这个
        #    - Transformer 模型的标准做法
        #
        # 注意：
        # - 这个 LayerNorm 的参数是独立的
        # - 不是任何 DecoderLayer 的内部 norm1、norm2 或 norm3
        self.norm = nn.LayerNorm(d_model)

        # 存储层数（供外部查询）
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码器前向传播

        参数：
            x: 目标序列输入，形状 (batch_size, tgt_len, d_model)
               （通常是目标的词嵌入 + 位置编码）

            encoder_output: 编码器输出，形状 (batch_size, src_len, d_model)
                           （编码后的源序列）

            tgt_mask: 目标掩码，形状 (batch_size, 1, tgt_len, tgt_len)
                     组合因果掩码 + 目标填充掩码

            src_mask: 源填充掩码，形状 (batch_size, 1, 1, src_len)
                     忽略源序列中的 <PAD> 词元

        返回：
            output: 解码器输出，形状 (batch_size, tgt_len, d_model)
                   （解码后的表示，准备用于最终投影）

        完整流程：
            输入 → 层1 → 层2 → ... → 层N → Norm → 输出

        具体例子（英语→法语翻译）：
            源：  "I love apples"
            目标："J'aime les pommes"

            假设 num_layers = 6, d_model = 512

            encoder_output:
                shape = (1, 3, 512)
                "I love apples" 的编码表示

            输入 x:
                shape = (1, 4, 512)
                x[0, 0, :] = "J'" 嵌入 + 位置编码
                x[0, 1, :] = "aime" 嵌入 + 位置编码
                x[0, 2, :] = "les" 嵌入 + 位置编码
                x[0, 3, :] = "pommes" 嵌入 + 位置编码

            层 1:
                - 掩码自注意力：从之前的法语词构建上下文
                - 交叉注意力：查看英语源
                - 前馈网络：提取基本特征
                → x.shape = (1, 4, 512)

            层 2:
                - 法语和英语之间更复杂的关系
                - 建立在层 1 的特征之上
                → x.shape = (1, 4, 512)

            层 3-6:
                - 逐步提取更抽象的特征
                - 最后一层包含最丰富的上下文信息
                → x.shape = (1, 4, 512)

            最终层归一化:
                - 归一化最终输出
                → x.shape = (1, 4, 512)

            输出:
                x[0, 0, :] = "J'" 编码（知道：这是开始，接下来是动词）
                x[0, 1, :] = "aime" 编码（知道：主语是 "I"，宾语即将到来）
                x[0, 2, :] = "les" 编码（知道：冠词，名词即将到来）
                x[0, 3, :] = "pommes" 编码（知道：这是宾语，结束句子）

                每个词的表示包含：
                ✓ 来自之前法语词的上下文（掩码自注意力）
                ✓ 来自英语源的信息（交叉注意力）
                ✓ 来自多层的丰富特征（深度）

        【为什么每层输出形状相同？】
        - 每层的输入和输出维度都是 d_model
        - 这允许：
          1. 使用残差连接（x + SubLayer(x)）
          2. 堆叠任意数量的层
          3. 灵活组合
          4. 所有层可以使用相同的 encoder_output

        【跨层的信息流】
        - 层 1 输出 → 成为层 2 输入
        - 层 2 输出 → 成为层 3 输入
        - ...
        - 最后一层包含所有之前层的信息
        - 每一层都添加对源-目标关系的更丰富理解
        """
        # ========== 顺序通过每个解码器层 ==========
        # for 循环按顺序执行：
        # x = layer_1(x, encoder_output, tgt_mask, src_mask)
        # x = layer_2(x, encoder_output, tgt_mask, src_mask)
        # x = layer_3(x, encoder_output, tgt_mask, src_mask)
        # ...
        # x = layer_N(x, encoder_output, tgt_mask, src_mask)
        #
        # 重要说明：
        # - 每层的输入是前一层的输出（x 被更新）
        # - encoder_output 保持不变（所有层相同）
        # - tgt_mask 保持不变（所有层相同的因果掩码）
        # - src_mask 保持不变（所有层相同的填充掩码）
        #
        # 为什么 encoder_output 不变？
        # - 编码器运行一次，产生源的固定表示
        # - 每个解码器层都使用这个相同的源表示
        # - 就像参考书：解码器不断查阅它，但书本身不变
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            # x.shape 始终为 (batch_size, tgt_len, d_model)

        # ========== 最终层归一化 ==========
        # 归一化最终输出
        # 确保输出分布稳定
        x = self.norm(x)

        # 最终输出：
        # - 形状：(batch_size, tgt_len, d_model)
        # - 与输入形状相同，但内容已被转换
        # - 每个目标词元的表示包含：
        #   * 来自之前目标词元的上下文（通过掩码自注意力）
        #   * 来自源序列的信息（通过交叉注意力）
        #   * 来自多层的丰富特征（通过深度）
        # - 这个输出可以：
        #   * 连接到输出投影（线性层到词汇表）
        #   * 用于下一个词元预测
        #   * 用于其他下游任务
        return x


def create_causal_mask(size: int) -> torch.Tensor:
    """
    为解码器自注意力创建因果掩码（也称为前瞻掩码）

    【这是什么？】
    一个下三角矩阵，防止位置注意到未来位置。
    用于解码器的掩码自注意力。

    【为什么需要？】
    在自回归生成过程中，我们一次生成一个词元：
    - 生成词元 3 时，我们只看到了词元 0, 1, 2
    - 我们不能看到词元 4, 5, 6...（它们还不存在！）
    - 在训练时，我们通过掩码未来位置来模拟这个

    【掩码格式】
    返回一个矩阵，其中：
    - 1 = 可以注意（位置可见）
    - 0 = 不能注意（位置被掩码）

    【示例】
    对于 size=4（有 4 个词元的句子）：

    [[1, 0, 0, 0],   ← 位置 0 只能看到位置 0
     [1, 1, 0, 0],   ← 位置 1 能看到位置 0-1
     [1, 1, 1, 0],   ← 位置 2 能看到位置 0-2
     [1, 1, 1, 1]]   ← 位置 3 能看到位置 0-3

    可视化表示：
    ```
           位置 0  1  2  3
    位置 0:  ✓  ✗  ✗  ✗
    位置 1:  ✓  ✓  ✗  ✗
    位置 2:  ✓  ✓  ✓  ✗
    位置 3:  ✓  ✓  ✓  ✓
    ```

    这被称为"因果"，因为：
    - 信息按因果顺序流动（过去 → 现在）
    - 不能倒流（未来 → 现在）
    - 尊重时间顺序

    参数：
        size: 序列长度（词元数）

    返回：
        mask: 因果掩码，形状 (1, 1, size, size)
              形状解释：
              - 第一个 1：批次维度（跨批次广播）
              - 第二个 1：头维度（跨注意力头广播）
              - (size, size)：实际掩码矩阵（query_len × key_len）

    在解码器中的用法：
        tgt_len = 5
        causal_mask = create_causal_mask(tgt_len)
        # causal_mask.shape = (1, 1, 5, 5)

        # 在注意力中使用：
        attention(query, key, value, mask=causal_mask)
        # 每个位置只能注意自己和之前的位置
    """
    # ========== 创建下三角矩阵 ==========
    # torch.tril 创建下三角矩阵
    # torch.ones(size, size) 创建全 1 矩阵
    # tril 将上三角设为 0，保持下三角为 1
    #
    # 示例 size=4：
    # torch.ones(4, 4):
    # [[1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1]]
    #
    # torch.tril(...):
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    #
    # 这正是我们想要的因果掩码！
    mask = torch.tril(torch.ones(size, size))
    # mask.shape = (size, size)

    # ========== 添加批次和头维度 ==========
    # 从 (size, size) 重塑为 (1, 1, size, size)
    # 为什么？
    # - 注意力期望掩码形状：(batch, heads, query_len, key_len)
    # - (1, 1, size, size) 会广播到 (batch, heads, size, size)
    #
    # 广播示例：
    # mask.shape = (1, 1, 4, 4)
    # attention scores.shape = (32, 8, 4, 4)  # batch=32, heads=8
    # → mask 自动广播到 (32, 8, 4, 4)
    #
    # .unsqueeze(0) 在位置 0 添加维度：(size, size) → (1, size, size)
    # .unsqueeze(0) 再次在位置 0 添加维度：(1, size, size) → (1, 1, size, size)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape = (1, 1, size, size)

    return mask


if __name__ == "__main__":
    # 测试代码
    print("=== 测试因果掩码 ===\n")

    # 创建一个小的因果掩码进行可视化
    causal_mask = create_causal_mask(5)
    print(f"因果掩码形状：{causal_mask.shape}")
    print("因果掩码 (5x5)：")
    print(causal_mask.squeeze().numpy())
    print("\n(1 = 能看到, 0 = 看不到)")

    print("\n=== 测试解码器层 ===\n")

    batch_size = 2
    src_len = 6  # 源序列长度（例如英语）
    tgt_len = 5  # 目标序列长度（例如法语）
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

    # 创建虚拟输入
    x = torch.randn(batch_size, tgt_len, d_model)  # 目标输入
    encoder_output = torch.randn(batch_size, src_len, d_model)  # 编码器输出

    print(f"目标输入形状：{x.shape}")
    print(f"编码器输出形状：{encoder_output.shape}")

    # 创建掩码
    tgt_mask = create_causal_mask(tgt_len)  # 目标的因果掩码
    src_mask = torch.ones(batch_size, 1, 1, src_len)  # 源没有填充（全为 1）

    # 前向传播
    output = decoder_layer(x, encoder_output, tgt_mask, src_mask)
    print(f"解码器层输出形状：{output.shape}")

    print("\n=== 测试完整解码器（6 层）===\n")

    num_layers = 6
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = decoder(x, encoder_output, tgt_mask, src_mask)
    print(f"完整解码器输出形状：{output_full.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n完整解码器（{num_layers} 层）总参数：{total_params:,}")
    print(f"大约 {total_params / 1e6:.1f}M 参数")
