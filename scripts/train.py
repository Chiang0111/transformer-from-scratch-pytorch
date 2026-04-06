"""
在序列任務上訓練 Transformer 模型

使用方式：
    # 複製任務（最簡單）
    python train.py --task copy --epochs 20

    # 反轉任務（中等難度）
    python train.py --task reverse --epochs 30

    # 排序任務（最困難）
    python train.py --task sort --epochs 50

    # 自訂配置
    python train.py --task copy --epochs 20 --batch-size 64 --d-model 256
"""

# 引入命令列參數解析模組，用於處理使用者輸入的訓練參數
import argparse
# 引入 PyTorch 深度學習框架的核心模組
import torch
# 引入 PyTorch 的神經網路模組，包含各種神經網路層和損失函數
import torch.nn as nn
# 引入路徑處理模組，用於跨平台的檔案路徑操作
from pathlib import Path
# 引入時間模組，用於計算訓練耗時
import time

# 從 transformer 模組引入建立 Transformer 模型的工廠函數
from transformer import create_transformer
# 從 datasets 模組引入建立資料載入器的函數，用於批次載入訓練資料
from datasets import create_dataloader
# 從 utils 模組引入多個訓練所需的工具函數和類別
from utils import (
    LabelSmoothingLoss,       # 標籤平滑損失函數，可防止模型過度自信
    TransformerLRScheduler,   # Transformer 專用的學習率調度器
    create_padding_mask,      # 建立填充遮罩，用於忽略填充符號
    create_target_mask,       # 建立目標遮罩，用於解碼器的因果注意力機制
    TrainingMetrics,          # 訓練指標追蹤器，記錄損失、準確率等
    save_checkpoint,          # 儲存模型檢查點的函數
    load_checkpoint,          # 載入模型檢查點的函數
    save_training_config      # 儲存訓練配置的函數
)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device: str,
    pad_idx: int,
    clip_grad: float = 1.0
) -> dict:
    """
    訓練一個完整的訓練週期（epoch）

    此函數會遍歷整個訓練資料集一次，對每個批次執行前向傳播、
    計算損失、反向傳播、更新權重等完整的訓練流程。

    參數：
        model: Transformer 模型實例，包含編碼器和解碼器
        train_loader: 訓練資料載入器，提供批次化的訓練資料
        criterion: 損失函數，用於計算模型預測與真實標籤之間的差距
        optimizer: 優化器，負責根據梯度更新模型參數
        scheduler: 學習率調度器，動態調整學習率以改善訓練效果
        device: 運算設備（'cpu' 或 'cuda'），決定在 CPU 或 GPU 上訓練
        pad_idx: 填充符號的索引值，用於建立遮罩以忽略填充位置
        clip_grad: 梯度裁剪閾值，防止梯度爆炸問題（預設為 1.0）

    回傳：
        包含訓練指標的字典，包括損失、準確率等統計資料
    """
    # 將模型設為訓練模式
    # 這會啟用 Dropout 和 Batch Normalization 等訓練時才需要的功能
    model.train()

    # 建立訓練指標追蹤器物件
    # 用於累積整個訓練週期的損失、準確率等統計資訊
    metrics = TrainingMetrics()

    # 遍歷訓練資料載入器中的所有批次
    # batch_idx: 當前批次的索引（從 0 開始）
    # src: 來源序列（輸入）形狀為 (batch_size, src_seq_len)
    # tgt_input: 目標輸入序列（解碼器的輸入）形狀為 (batch_size, tgt_seq_len)
    # tgt_output: 目標輸出序列（用於計算損失）形狀為 (batch_size, tgt_seq_len)
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        # ====== 資料準備階段 ======
        # 將來源序列移動到指定的運算設備（CPU 或 GPU）
        # 這確保所有張量都在同一個設備上，避免運算錯誤
        src = src.to(device)
        # 將目標輸入序列移動到運算設備
        tgt_input = tgt_input.to(device)
        # 將目標輸出序列移動到運算設備
        tgt_output = tgt_output.to(device)

        # ====== 建立注意力遮罩 ======
        # 建立來源序列的填充遮罩
        # 目的：讓模型忽略填充符號（pad），不對其進行注意力計算
        # 形狀為 (batch_size, 1, 1, src_seq_len)
        src_mask = create_padding_mask(src, pad_idx).to(device)

        # 建立目標序列的組合遮罩（填充遮罩 + 因果遮罩）
        # 填充遮罩：忽略填充符號
        # 因果遮罩：確保解碼器在位置 i 只能看到位置 < i 的資訊（防止看到未來）
        # 形狀為 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)

        # ====== 前向傳播階段 ======
        # 將輸入資料送入 Transformer 模型進行前向傳播
        # src: 編碼器的輸入序列
        # tgt_input: 解碼器的輸入序列
        # src_mask: 編碼器的注意力遮罩
        # tgt_mask: 解碼器的自注意力遮罩
        # logits: 模型的原始輸出（未經 softmax），形狀為 (batch_size, tgt_seq_len, vocab_size)
        # 每個位置都有一個對整個詞彙表的評分向量
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # ====== 損失計算階段 ======
        # 使用損失函數計算模型預測與真實標籤之間的差距
        # logits: 模型的原始輸出，包含每個詞彙的預測分數
        # tgt_output: 真實的目標序列，包含正確的詞彙索引
        # 損失函數會自動處理 softmax 和交叉熵的計算
        loss = criterion(logits, tgt_output)

        # ====== 反向傳播階段 ======
        # 清空優化器中所有參數的梯度
        # PyTorch 預設會累積梯度，所以每次反向傳播前必須清零
        # 否則梯度會不斷累加，導致訓練錯誤
        optimizer.zero_grad()

        # 執行反向傳播，計算損失對所有模型參數的梯度
        # 這會自動使用鏈式法則計算整個計算圖的梯度
        # 梯度會儲存在每個參數的 .grad 屬性中
        loss.backward()

        # ====== 梯度裁剪 ======
        # 對所有模型參數的梯度進行範數裁剪，防止梯度爆炸
        # 當梯度的 L2 範數超過 clip_grad 時，會按比例縮小梯度
        # 這對於訓練 RNN、Transformer 等深度序列模型非常重要
        # 可以穩定訓練過程，避免數值不穩定
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # ====== 參數更新階段 ======
        # 使用優化器根據計算出的梯度更新模型參數
        # 具體更新規則取決於優化器類型（Adam、SGD 等）
        # Adam 會使用動量和自適應學習率來更新參數
        optimizer.step()

        # ====== 學習率調整 ======
        # 如果有提供學習率調度器，則更新學習率
        # Transformer 論文使用特殊的 warmup + decay 調度策略
        # 先線性增加學習率（warmup），然後按照特定公式衰減
        if scheduler is not None:
            scheduler.step()

        # ====== 指標更新 ======
        # 在不計算梯度的模式下更新訓練指標
        # torch.no_grad() 可以節省記憶體並加快計算速度
        # 因為我們不需要為指標計算保留梯度資訊
        with torch.no_grad():
            # 更新累積指標：損失、符號準確率、序列準確率等
            # loss.item(): 將損失張量轉換為 Python 標量值
            # logits: 模型的預測輸出
            # tgt_output: 真實標籤
            # pad_idx: 填充符號索引，計算準確率時會忽略
            metrics.update(loss.item(), logits, tgt_output, pad_idx)

        # ====== 進度顯示 ======
        # 每 50 個批次印出一次訓練進度
        # 這有助於監控訓練狀況，而不會產生過多輸出
        if (batch_idx + 1) % 50 == 0:
            # 獲取目前累積的平均指標
            current_metrics = metrics.get_metrics()

            # 獲取目前的學習率
            # 如果使用調度器，從調度器取得；否則從優化器取得
            if scheduler is not None:
                # 調度器記錄的最後一次學習率
                lr = scheduler.get_last_lr()
            else:
                # 從優化器的參數組中取得學習率
                # param_groups[0] 是第一個參數組（通常只有一個）
                lr = optimizer.param_groups[0]['lr']

            # 印出訓練進度資訊
            # 包括：批次進度、損失、符號準確率、序列準確率、學習率
            # .4f 表示保留 4 位小數，.2f 表示保留 2 位小數，.6f 表示保留 6 位小數
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {current_metrics['loss']:.4f} | "
                  f"Token Acc: {current_metrics['token_accuracy']:.2f}% | "
                  f"Seq Acc: {current_metrics['sequence_accuracy']:.2f}% | "
                  f"LR: {lr:.6f}")

    # 回傳整個訓練週期的平均指標
    # 包含損失、符號準確率、序列準確率、困惑度等
    return metrics.get_metrics()


@torch.no_grad()  # 裝飾器：在此函數內停用梯度計算，節省記憶體並加速運算
def evaluate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
    pad_idx: int
) -> dict:
    """
    在驗證集上評估模型效能

    此函數會在驗證資料集上執行模型推論，計算損失和準確率等指標。
    與訓練不同的是，這裡不會執行反向傳播和參數更新，
    而且會使用 torch.no_grad() 來停用梯度計算以節省記憶體。

    參數：
        model: Transformer 模型實例，要評估的模型
        val_loader: 驗證資料載入器，提供批次化的驗證資料
        criterion: 損失函數，用於計算驗證損失
        device: 運算設備（'cpu' 或 'cuda'）
        pad_idx: 填充符號的索引值，計算指標時會忽略此符號

    回傳：
        包含驗證指標的字典，包括損失、準確率、困惑度等
    """
    # 將模型設為評估模式
    # 這會停用 Dropout（不再隨機丟棄神經元）和 Batch Normalization 的訓練行為
    # 確保推論結果具有確定性和穩定性
    model.eval()

    # 建立指標追蹤器物件，用於累積整個驗證集的統計資料
    metrics = TrainingMetrics()

    # 遍歷驗證資料載入器中的所有批次
    # 注意：這裡不需要 batch_idx，因為驗證時不需要印出進度
    # src: 來源序列（形狀：batch_size × src_seq_len）
    # tgt_input: 目標輸入序列（形狀：batch_size × tgt_seq_len）
    # tgt_output: 目標輸出序列（形狀：batch_size × tgt_seq_len）
    for src, tgt_input, tgt_output in val_loader:
        # ====== 資料準備階段 ======
        # 將來源序列移動到運算設備（CPU 或 GPU）
        src = src.to(device)
        # 將目標輸入序列移動到運算設備
        tgt_input = tgt_input.to(device)
        # 將目標輸出序列移動到運算設備
        tgt_output = tgt_output.to(device)

        # ====== 建立注意力遮罩 ======
        # 建立來源序列的填充遮罩
        # 讓模型在注意力計算時忽略填充符號
        src_mask = create_padding_mask(src, pad_idx).to(device)

        # 建立目標序列的組合遮罩（填充遮罩 + 因果遮罩）
        # 確保解碼器只能看到當前位置之前的資訊
        tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)

        # ====== 前向傳播階段 ======
        # 將資料送入模型進行推論
        # 由於使用了 @torch.no_grad() 裝飾器，此過程不會建立計算圖
        # logits: 模型的原始輸出（未經 softmax）
        # 形狀為 (batch_size, tgt_seq_len, vocab_size)
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # ====== 損失計算 ======
        # 計算預測輸出與真實標籤之間的損失
        # 用於評估模型在驗證集上的效能
        # 注意：這裡只是計算損失，不會進行反向傳播
        loss = criterion(logits, tgt_output)

        # ====== 更新評估指標 ======
        # 累積當前批次的統計資料到指標追蹤器
        # loss.item(): 將損失張量轉換為純 Python 數值
        # logits: 用於計算預測準確率
        # tgt_output: 真實標籤，用於比較
        # pad_idx: 忽略填充符號，不計入準確率統計
        metrics.update(loss.item(), logits, tgt_output, pad_idx)

    # 回傳整個驗證集的平均指標
    # 包含損失（loss）、符號準確率（token_accuracy）、
    # 序列準確率（sequence_accuracy）、困惑度（perplexity）等
    return metrics.get_metrics()


@torch.no_grad()  # 裝飾器：停用梯度計算，因為這是測試階段不需要訓練
def test_generation(
    model: nn.Module,
    test_samples: list,
    device: str,
    start_token: int,
    end_token: int,
    max_len: int = 20
):
    """
    測試自回歸生成能力

    【功能說明】
    此函數使用自回歸（autoregressive）方式測試模型的生成能力。
    自回歸是指模型會逐個符號地生成輸出序列，每次生成時都會
    參考之前已生成的所有符號。這能讓我們直觀地看到模型是否
    真正學會了任務，而不只是在訓練集上記憶答案。

    【為什麼需要這個測試】
    - 驗證損失和準確率只能看到數字，無法看到實際預測結果
    - 透過實際範例可以快速發現模型的問題（如總是預測相同的值）
    - 有助於理解模型的學習進度和能力

    參數：
        model: Transformer 模型實例
        test_samples: 測試樣本列表，每個元素是 (來源序列, 預期輸出) 的元組
        device: 運算設備（'cpu' 或 'cuda'）
        start_token: 開始符號的 ID（通常是 1），解碼器生成的起始符號
        end_token: 結束符號的 ID（通常是 2），表示序列生成完成
        max_len: 最大生成長度，防止模型無限生成（預設為 20）
    """
    # 將模型設為評估模式
    # 停用 Dropout 等訓練時的隨機性，確保生成結果穩定可重現
    model.eval()

    # 印出測試標題區塊
    # 使用分隔線讓輸出更清晰易讀
    print("\n" + "="*60)
    print(">> 生成測試 - 看看模型學到了什麼！")
    print("="*60)

    # 遍歷所有測試樣本
    # enumerate(test_samples, 1): 從 1 開始編號（而非預設的 0）
    # i: 樣本編號（1, 2, 3...）
    # src: 來源序列（輸入）
    # expected: 預期的輸出序列
    for i, (src, expected) in enumerate(test_samples, 1):
        # ====== 準備輸入 ======
        # 為來源序列增加批次維度
        # src 原本是 1D 張量 (seq_len,)
        # unsqueeze(0) 將其變成 2D 張量 (1, seq_len)，表示批次大小為 1
        # 因為模型期望輸入的第一個維度是批次大小
        src_batch = src.unsqueeze(0).to(device)

        # ====== 自回歸生成 ======
        # 使用模型的 generate 方法進行自回歸生成
        # 過程：從 start_token 開始，逐步生成每個符號
        # 每次生成時，都會將已生成的序列作為解碼器的輸入
        # 直到生成 end_token 或達到最大長度 max_len
        # 回傳：完整的生成序列（包含特殊符號）
        generated = model.generate(
            src_batch,              # 編碼器的輸入（批次大小為 1）
            max_len=max_len,        # 最大生成長度（防止無限生成）
            start_token=start_token,  # 解碼器的起始符號
            end_token=end_token      # 生成結束的標記符號
        )

        # ====== 後處理生成結果 ======
        # 移除批次維度並轉換為 Python 列表
        # squeeze(0): 將 (1, seq_len) 變回 (seq_len,)
        # cpu(): 將張量從 GPU 移回 CPU（如果在 GPU 上）
        # tolist(): 將張量轉換為 Python 列表，方便顯示和比較
        generated = generated.squeeze(0).cpu().tolist()

        # ====== 清理特殊符號 ======
        # 移除特殊符號以便顯示
        # 0: 填充符號（PAD）
        # start_token: 開始符號（通常是 1）
        # end_token: 結束符號（通常是 2）
        # 這些符號對人類閱讀沒有意義，所以過濾掉

        # 清理輸入序列：移除所有特殊符號
        # 列表推導式：只保留不是特殊符號的元素
        src_clean = [x for x in src.tolist() if x not in [0, start_token, end_token]]

        # 清理預期輸出：移除所有特殊符號
        expected_clean = [x for x in expected.tolist() if x not in [0, start_token, end_token]]

        # 清理模型生成的輸出：移除所有特殊符號
        generated_clean = [x for x in generated if x not in [0, start_token, end_token]]

        # ====== 檢查正確性 ======
        # 比較模型生成的結果與預期輸出是否完全一致
        # 這是序列層級的準確率判斷：必須所有符號都正確才算對
        is_correct = generated_clean == expected_clean

        # 根據正確性設定狀態標籤
        # [OK] CORRECT: 完全正確
        # [X] WRONG: 有錯誤
        status = "[OK] CORRECT" if is_correct else "[X] WRONG"

        # ====== 顯示結果 ======
        # 印出這個測試樣本的詳細資訊
        print(f"\n範例 {i}: {status}")
        print(f"  輸入：    {src_clean}")
        print(f"  預期：    {expected_clean}")
        print(f"  模型生成：{generated_clean}")

    # 印出結束分隔線
    print("="*60 + "\n")


def main():
    # ====== 解析命令列參數 ======
    # 建立命令列參數解析器
    # 讓使用者可以透過命令列指定各種訓練參數（如任務類型、訓練週期數等）
    parser = argparse.ArgumentParser(description='在序列任務上訓練 Transformer 模型')

    # ====== 任務相關設定 ======
    # 指定要訓練的任務類型
    # copy: 複製任務（最簡單），將輸入序列原樣輸出
    # reverse: 反轉任務（中等難度），將輸入序列反轉
    # sort: 排序任務（最困難），將輸入序列由小到大排序
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='要訓練的任務類型')

    # 訓練樣本數量
    # 預設 10000 個樣本，對於簡單任務已經足夠
    # 更複雜的任務可能需要更多樣本
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='訓練樣本數量')

    # 詞彙表大小
    # 預設 20 個符號（0: PAD, 1: START, 2: END, 3-19: 實際符號）
    # 詞彙表越大，任務越困難
    parser.add_argument('--vocab-size', type=int, default=20,
                        help='詞彙表大小')

    # ====== 模型架構設定 ======
    # d_model: 模型的維度（嵌入向量和隱藏層的維度）
    # 預設 128，這是 Transformer 內部所有向量的維度
    # 更大的值可以提升模型容量，但會增加計算成本和記憶體需求
    parser.add_argument('--d-model', type=int, default=128,
                        help='模型維度（嵌入向量和隱藏層的維度）')

    # num_heads: 多頭注意力機制的頭數
    # 預設 4 個頭，每個頭會學習不同的注意力模式
    # 必須能整除 d_model（例如 d_model=128, num_heads=4, 每個頭的維度=32）
    parser.add_argument('--num-heads', type=int, default=4,
                        help='多頭注意力的頭數')

    # num_layers: 編碼器和解碼器的層數
    # 預設各 2 層（編碼器 2 層 + 解碼器 2 層）
    # 更多層可以學習更複雜的模式，但也更難訓練
    parser.add_argument('--num-layers', type=int, default=2,
                        help='編碼器和解碼器的層數')

    # d_ff: 前饋神經網路的隱藏層維度
    # 預設 512，通常設為 d_model 的 4 倍
    # Transformer 論文使用 d_model=512, d_ff=2048
    parser.add_argument('--d-ff', type=int, default=512,
                        help='前饋神經網路的隱藏層維度')

    # dropout: Dropout 機率
    # 預設 0.1，表示隨機丟棄 10% 的神經元
    # Dropout 是一種正則化技術，可以防止過擬合
    # 評估時會自動停用
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 機率（防止過擬合）')

    # ====== 訓練過程設定 ======
    # epochs: 訓練週期數
    # 一個 epoch 表示遍歷整個訓練集一次
    # 不同任務需要不同的訓練週期數（copy: 20, reverse: 30, sort: 50）
    parser.add_argument('--epochs', type=int, default=20,
                        help='訓練週期數（遍歷訓練集的次數）')

    # batch_size: 批次大小
    # 每次訓練時同時處理的樣本數量
    # 較大的批次可以加速訓練但需要更多記憶體
    # 較小的批次提供更多梯度更新但可能不穩定
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小（每次訓練的樣本數）')

    # fixed_lr: 固定學習率
    # 如果設定此參數，將使用固定學習率而非 Transformer 的調度策略
    # 對於小模型和簡單任務，固定學習率通常效果更好
    # 例如：--fixed-lr 0.001
    parser.add_argument('--fixed-lr', type=float, default=None,
                        help='使用固定學習率（例如 0.001），而非 Transformer 調度策略')

    # lr_factor: 學習率因子（僅用於 Transformer 調度）
    # 預設 2.0，控制學習率的整體大小
    # 只有在不使用 --fixed-lr 時才會生效
    parser.add_argument('--lr-factor', type=float, default=2.0,
                        help='學習率因子（僅用於 Transformer 調度策略）')

    # warmup_steps: 預熱步數（僅用於 Transformer 調度）
    # 預設 1000 步，在這些步數內學習率會線性增加
    # 之後會按照特定公式衰減
    # 預熱可以穩定訓練初期的梯度
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='預熱步數（僅用於 Transformer 調度策略）')

    # label_smoothing: 標籤平滑係數
    # 預設 0.1，將硬標籤（0 或 1）軟化為機率分布
    # 例如：原本 [0, 1, 0] 變成 [0.05, 0.9, 0.05]
    # 可以防止模型過度自信，改善泛化能力
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='標籤平滑係數（防止過度自信）')

    # clip_grad: 梯度裁剪閾值
    # 預設 1.0，當梯度的 L2 範數超過此值時會按比例縮小
    # 這可以防止梯度爆炸，穩定訓練過程
    # 對於 RNN、LSTM、Transformer 等序列模型特別重要
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='梯度裁剪閾值（防止梯度爆炸）')

    # ====== 其他設定 ======
    # device: 運算設備
    # 'cpu': 使用 CPU 運算（較慢但相容性好）
    # 'cuda': 使用 GPU 運算（快很多，需要 NVIDIA GPU）
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='運算設備（cpu 或 cuda）')

    # checkpoint_dir: 檢查點儲存目錄
    # 訓練過程中會定期儲存模型檢查點到此目錄
    # 可以用於恢復訓練或載入最佳模型
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='模型檢查點儲存目錄')

    # resume: 恢復訓練的檢查點路徑
    # 如果提供，會從此檢查點恢復訓練（載入模型權重、優化器狀態等）
    # 可以用於中斷後繼續訓練
    parser.add_argument('--resume', type=str, default=None,
                        help='要恢復訓練的檢查點路徑')

    # 解析使用者提供的命令列參數
    # 將所有參數儲存在 args 物件中，可以透過 args.參數名稱 存取
    # 例如：args.task, args.epochs, args.d_model 等
    args = parser.parse_args()

    # ====== 設定運算設備 ======
    # 從參數中取得使用者指定的設備
    device = args.device

    # 檢查 CUDA（GPU）是否可用
    # 如果使用者指定使用 cuda 但系統沒有 GPU，則退回到 CPU
    # torch.cuda.is_available() 會檢查是否有可用的 NVIDIA GPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，改用 CPU")
        device = 'cpu'

    # ====== 印出訓練資訊標題 ======
    # 顯示訓練配置的摘要資訊
    print("\n" + "="*60)
    print(">> TRANSFORMER 訓練")
    print("="*60)
    # 顯示任務類型（全部大寫）
    print(f"任務：{args.task.upper()}")
    # 顯示使用的運算設備
    print(f"設備：{device.upper()}")
    # 顯示模型的關鍵參數：模型維度、層數、注意力頭數
    print(f"模型：d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print("="*60 + "\n")

    # ====== 建立資料載入器 ======
    print("[*] 載入資料...")
    # 建立訓練、驗證、測試三個資料載入器
    # train_loader: 訓練資料載入器，用於訓練模型
    # val_loader: 驗證資料載入器，用於調整超參數和監控訓練進度
    # test_loader: 測試資料載入器，用於最終評估（從未見過的資料）
    # dataset_info: 資料集的詳細資訊（任務類型、樣本數、詞彙表大小等）
    train_loader, val_loader, test_loader, dataset_info = create_dataloader(
        dataset_type=args.task,      # 任務類型（copy/reverse/sort）
        batch_size=args.batch_size,  # 批次大小（一次處理多少個樣本）
        num_samples=args.num_samples,  # 總樣本數
        vocab_size=args.vocab_size   # 詞彙表大小（符號種類數）
    )

    # 印出資料集的詳細資訊
    print(f"[+] 資料集準備完成：")
    # 任務類型（copy/reverse/sort）
    print(f"   任務：{dataset_info['task']}")
    # 訓練樣本數量（通常佔總樣本的 80%）
    print(f"   訓練樣本數：{dataset_info['train_samples']}")
    # 驗證樣本數量（通常佔總樣本的 10%）
    print(f"   驗證樣本數：{dataset_info['val_samples']}")
    # 測試樣本數量（通常佔總樣本的 10%）
    print(f"   測試樣本數：{dataset_info['test_samples']}")
    # 詞彙表大小（包含特殊符號 PAD、START、END）
    print(f"   詞彙表大小：{dataset_info['vocab_size']}")
    print()

    # ====== 建立模型 ======
    print("[*] 建立模型...")
    # 使用工廠函數建立 Transformer 模型
    # 包含編碼器（Encoder）和解碼器（Decoder）
    model = create_transformer(
        src_vocab_size=dataset_info['vocab_size'],  # 來源詞彙表大小（輸入）
        tgt_vocab_size=dataset_info['vocab_size'],  # 目標詞彙表大小（輸出）
        d_model=args.d_model,        # 模型維度（嵌入和隱藏層的維度）
        num_heads=args.num_heads,    # 多頭注意力的頭數
        num_layers=args.num_layers,  # 編碼器和解碼器的層數
        d_ff=args.d_ff,             # 前饋網路的隱藏層維度
        dropout=args.dropout        # Dropout 機率（防止過擬合）
    )

    # 將模型移動到指定的運算設備（CPU 或 GPU）
    # 這確保模型的所有參數都在正確的設備上
    # 必須在訓練前完成，否則會出現設備不匹配錯誤
    model = model.to(device)

    # 計算模型的總參數數量
    # count_parameters() 是模型的方法，會統計所有可訓練參數
    param_count = model.count_parameters()

    # 印出模型資訊
    print(f"[+] 模型建立完成：")
    # 參數數量（使用千位分隔符，例如 1,234,567）
    print(f"   參數數量：{param_count:,}")
    # 模型大小估計（MB）
    # 假設每個參數佔 4 bytes（32-bit float）
    # 除以 1024 兩次將 bytes 轉換為 MB
    print(f"   大小：約 {param_count * 4 / 1024 / 1024:.2f} MB")
    print()

    # ====== 建立優化器和學習率調度器 ======
    if args.fixed_lr is not None:
        # 使用固定學習率（適合小模型和簡單任務）
        # Adam 優化器：結合動量（momentum）和自適應學習率
        # betas=(0.9, 0.999): 一階和二階動量的衰減率
        # eps=1e-9: 防止除以零的小常數
        optimizer = torch.optim.Adam(
            model.parameters(),  # 要優化的參數（模型的所有權重）
            lr=args.fixed_lr,    # 學習率（控制參數更新的步長）
            betas=(0.9, 0.999),  # 動量參數（平滑梯度更新）
            eps=1e-9            # 數值穩定性參數
        )
        # 不使用學習率調度器（固定學習率）
        scheduler = None
        print(f"[*] 使用固定學習率：{args.fixed_lr}")
    else:
        # 使用 Transformer 論文中的學習率調度策略
        # 這種策略包含預熱（warmup）和衰減兩個階段
        # 預熱階段：學習率線性增加，穩定訓練初期
        # 衰減階段：學習率按照公式逐漸降低
        optimizer = torch.optim.Adam(
            model.parameters(),  # 要優化的參數
            lr=1.0,             # 初始學習率（會被調度器調整，這裡的值不重要）
            betas=(0.9, 0.98),  # 論文中使用的動量參數（與固定學習率略有不同）
            eps=1e-9           # 數值穩定性參數
        )

        # 建立 Transformer 學習率調度器
        # 實作論文中的公式：lr = factor * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
        scheduler = TransformerLRScheduler(
            optimizer,                    # 要控制的優化器
            d_model=args.d_model,        # 模型維度（影響學習率大小）
            warmup_steps=args.warmup_steps,  # 預熱步數
            factor=args.lr_factor        # 學習率縮放因子（調整整體大小）
        )
        print(f"[*] 使用 Transformer 學習率調度：factor={args.lr_factor}, warmup={args.warmup_steps}")

    # ====== 建立損失函數 ======
    # 使用標籤平滑損失函數（Label Smoothing Loss）
    # 標籤平滑可以防止模型過度自信，改善泛化能力
    # 例如：原本硬標籤 [0, 1, 0] 變成 [0.05, 0.9, 0.05]
    criterion = LabelSmoothingLoss(
        smoothing=args.label_smoothing,  # 平滑係數（0 表示不平滑，>0 表示平滑）
        pad_idx=dataset_info['pad_token']  # 填充符號索引（計算損失時忽略）
    )

    # ====== 載入檢查點（如果要恢復訓練）======
    # 起始訓練週期，預設從 0 開始
    start_epoch = 0

    # 如果使用者提供了恢復訓練的檢查點路徑
    if args.resume:
        # 載入檢查點，恢復模型權重、優化器狀態、調度器狀態
        # 這讓我們可以從中斷的地方繼續訓練，而不必重新開始
        checkpoint_info = load_checkpoint(
            args.resume,   # 檢查點檔案路徑
            model,         # 模型物件（會載入權重）
            optimizer,     # 優化器物件（會載入狀態，如動量）
            scheduler      # 調度器物件（會載入狀態，如目前步數）
        )

        # 從檢查點中取得上次訓練到的週期數
        # +1 是因為要從下一個週期開始
        # 例如：如果檢查點是 epoch 9，則從 epoch 10 開始
        start_epoch = checkpoint_info['epoch'] + 1
        print(f"[*] 從週期 {start_epoch} 恢復訓練")
        print()

    # ====== 儲存訓練配置 ======
    # 將所有訓練配置儲存為 JSON 檔案
    # 這對於之後重現實驗或載入模型時非常有用
    config = {
        'task': args.task,  # 任務類型
        'model': {          # 模型架構參數
            'd_model': args.d_model,        # 模型維度
            'num_heads': args.num_heads,    # 注意力頭數
            'num_layers': args.num_layers,  # 層數
            'd_ff': args.d_ff,             # 前饋網路維度
            'dropout': args.dropout,        # Dropout 機率
            'vocab_size': args.vocab_size   # 詞彙表大小
        },
        'training': {       # 訓練超參數
            'epochs': args.epochs,                # 訓練週期數
            'batch_size': args.batch_size,        # 批次大小
            'lr_factor': args.lr_factor,          # 學習率因子
            'warmup_steps': args.warmup_steps,    # 預熱步數
            'label_smoothing': args.label_smoothing,  # 標籤平滑
            'clip_grad': args.clip_grad           # 梯度裁剪閾值
        }
    }

    # 將配置字典儲存為 JSON 檔案到檢查點目錄
    # 檔名通常是 config.json
    save_training_config(config, args.checkpoint_dir)

    # ====== 準備測試樣本用於生成測試 ======
    # 建立一個空列表來儲存測試樣本
    test_samples = []

    # 取得底層資料集物件
    # train_loader 是 DataLoader，train_loader.dataset 是 Subset
    # train_loader.dataset.dataset 是實際的資料集物件
    dataset_obj = train_loader.dataset.dataset

    # 從資料集中取得前 5 個樣本用於生成測試
    # 這些樣本會在訓練過程中定期用來展示模型的生成能力
    for i in range(5):
        # 取得第 i 個樣本
        # src: 來源序列（輸入）
        # _: 目標輸入（不需要，用 _ 忽略）
        # tgt_out: 目標輸出（預期答案）
        src, _, tgt_out = dataset_obj[i]
        # 將 (輸入, 預期輸出) 元組加入測試樣本列表
        test_samples.append((src, tgt_out))

    # ====== 開始訓練迴圈 ======
    print("[*] 開始訓練...")
    print("="*60 + "\n")

    # 記錄最佳驗證準確率
    # 用於判斷是否需要儲存最佳模型
    best_val_acc = 0.0

    # 主訓練迴圈：遍歷所有訓練週期
    # range(start_epoch, args.epochs): 從 start_epoch 到 args.epochs-1
    # 如果是新訓練，start_epoch=0；如果是恢復訓練，則從上次中斷處開始
    for epoch in range(start_epoch, args.epochs):
        # 記錄這個週期的開始時間
        # 用於計算每個週期的訓練時長
        epoch_start = time.time()

        # 印出目前週期資訊
        # epoch + 1: 因為從 0 開始計數，顯示時 +1 更符合人類習慣
        # 例如：epoch=0 顯示為 "Epoch 1/20"
        print(f"週期 {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # ====== 訓練階段 ======
        # 在訓練集上訓練一個完整週期
        # 回傳：包含損失、準確率等指標的字典
        train_metrics = train_epoch(
            model,                          # Transformer 模型
            train_loader,                   # 訓練資料載入器
            criterion,                      # 損失函數
            optimizer,                      # 優化器
            scheduler,                      # 學習率調度器
            device,                         # 運算設備
            dataset_info['pad_token'],      # 填充符號索引
            args.clip_grad                  # 梯度裁剪閾值
        )

        # ====== 驗證階段 ======
        # 在驗證集上評估模型效能
        # 不會更新模型參數，只是評估
        # 回傳：包含損失、準確率等指標的字典
        val_metrics = evaluate(
            model,                          # Transformer 模型
            val_loader,                     # 驗證資料載入器
            criterion,                      # 損失函數
            device,                         # 運算設備
            dataset_info['pad_token']       # 填充符號索引
        )

        # 計算這個週期的訓練時長（秒）
        # time.time() - epoch_start: 當前時間減去開始時間
        epoch_time = time.time() - epoch_start

        # ====== 印出週期摘要 ======
        print(f"\n[>] 週期 {epoch + 1} 摘要：")

        # 印出訓練集的指標
        # .4f: 保留 4 位小數
        # .2f: 保留 2 位小數
        print(f"   訓練損失：{train_metrics['loss']:.4f} | "
              f"符號準確率：{train_metrics['token_accuracy']:.2f}% | "
              f"序列準確率：{train_metrics['sequence_accuracy']:.2f}%")

        # 印出驗證集的指標
        # 驗證指標更重要，因為能反映模型的泛化能力
        print(f"   驗證損失：{val_metrics['loss']:.4f} | "
              f"符號準確率：{val_metrics['token_accuracy']:.2f}% | "
              f"序列準確率：{val_metrics['sequence_accuracy']:.2f}%")

        # 印出這個週期的訓練時長
        print(f"   時間：{epoch_time:.1f}秒")

        # ====== 儲存檢查點 ======
        # 在以下兩種情況下儲存檢查點：
        # 1. 每 5 個週期（定期儲存）
        # 2. 驗證準確率超過目前最佳（儲存最佳模型）
        if (epoch + 1) % 5 == 0 or val_metrics['sequence_accuracy'] > best_val_acc:
            # 儲存檢查點（包含模型權重、優化器狀態、調度器狀態、指標等）
            save_checkpoint(
                model,                      # 模型物件
                optimizer,                  # 優化器物件
                scheduler,                  # 調度器物件
                epoch,                      # 目前週期數
                {'train': train_metrics, 'val': val_metrics},  # 訓練和驗證指標
                args.checkpoint_dir        # 儲存目錄
            )

            # 如果這是目前最佳模型（驗證準確率最高）
            if val_metrics['sequence_accuracy'] > best_val_acc:
                # 更新最佳驗證準確率
                best_val_acc = val_metrics['sequence_accuracy']

                # 另外儲存一份為最佳模型
                # 使用特殊檔名 'checkpoint_best.pt'
                # 這樣之後可以輕鬆載入表現最好的模型
                save_checkpoint(
                    model,                      # 模型物件
                    optimizer,                  # 優化器物件
                    scheduler,                  # 調度器物件
                    epoch,                      # 目前週期數
                    {'train': train_metrics, 'val': val_metrics},  # 指標
                    args.checkpoint_dir,        # 儲存目錄
                    filename='checkpoint_best.pt'  # 指定檔名為最佳模型
                )
                print(f"   [!] 新的最佳模型！驗證序列準確率：{best_val_acc:.2f}%")

        # ====== 生成測試 ======
        # 每 5 個週期執行一次生成測試
        # 讓我們可以直觀地看到模型的學習進度
        # 而不只是看數字指標
        if (epoch + 1) % 5 == 0:
            test_generation(
                model,                          # 模型物件
                test_samples,                   # 測試樣本列表
                device,                         # 運算設備
                dataset_info['start_token'],    # 開始符號
                dataset_info['end_token']       # 結束符號
            )

        # 印出空行，讓輸出更清晰
        print()

    # ====== 最終測試集評估 ======
    # 測試集是從未在訓練過程中見過的資料（held-out data）
    # 這是評估模型真正泛化能力的最重要指標
    print("\n" + "="*60)
    print(">> 訓練完成！")
    print("="*60)
    print(f"最佳驗證序列準確率：{best_val_acc:.2f}%\n")

    # 在測試集上進行最終評估
    # 測試集的效能反映了模型在真實世界未見資料上的表現
    print("="*60)
    print(">> 最終測試集評估")
    print("="*60)
    print("在留存測試集上評估（訓練期間從未見過）...\n")

    # 使用 evaluate 函數計算測試集的指標
    # 這與驗證過程相同，但使用的是完全獨立的測試資料
    test_metrics = evaluate(
        model,                          # 訓練好的模型
        test_loader,                    # 測試資料載入器
        criterion,                      # 損失函數
        device,                         # 運算設備
        dataset_info['pad_token']       # 填充符號索引
    )

    # 印出測試集的詳細結果
    print(f"[+] 測試集結果：")
    # 損失值：數值越低越好
    print(f"   損失：     {test_metrics['loss']:.4f}")
    # 符號準確率：單個符號的預測準確率（百分比）
    print(f"   符號準確率：{test_metrics['token_accuracy']:.2f}%")
    # 序列準確率：整個序列完全正確的比例（百分比）
    # 這是最重要的指標，因為序列任務需要所有符號都正確
    print(f"   序列準確率：{test_metrics['sequence_accuracy']:.2f}%")
    # 困惑度（Perplexity）：越低表示模型越確定
    # 是損失的指數形式，更直觀地反映模型的不確定性
    print(f"   困惑度：    {test_metrics['perplexity']:.2f}")
    print("="*60 + "\n")

    # ====== 最終生成測試 ======
    # 展示一些具體的生成範例
    # 讓我們可以直觀地看到模型學到了什麼
    test_generation(
        model,                          # 訓練好的模型
        test_samples,                   # 測試樣本
        device,                         # 運算設備
        dataset_info['start_token'],    # 開始符號
        dataset_info['end_token']       # 結束符號
    )

    # ====== 印出後續使用說明 ======
    # 告訴使用者檢查點儲存在哪裡
    print("[+] 模型檢查點已儲存至：", args.checkpoint_dir)

    # 提供測試模型的命令範例
    print("\n如何測試模型：")
    print(f"  python test.py --checkpoint {args.checkpoint_dir}/checkpoint_best.pt --task {args.task}")


# 主程式入口點
# 當這個檔案直接執行時（而非被匯入時），會執行 main() 函數
if __name__ == '__main__':
    main()
