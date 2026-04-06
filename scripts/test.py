"""
測試已訓練的 Transformer 模型

使用方式：
    python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy
"""

# 引入命令列參數解析模組，處理使用者輸入的測試參數
import argparse
# 引入 PyTorch 深度學習框架
import torch
# 從 transformer 模組引入建立模型的工廠函數
from transformer import create_transformer
# 從 datasets 模組引入建立資料載入器的函數
from datasets import create_dataloader
# 從 utils 模組引入多個工具函數
from utils import load_checkpoint, create_padding_mask, create_target_mask, TrainingMetrics
# 引入 JSON 模組，用於讀取配置檔案
import json
# 引入路徑處理模組，用於處理檔案路徑
from pathlib import Path


@torch.no_grad()  # 裝飾器：停用梯度計算，因為測試階段不需要訓練
def test_model(model, test_loader, device, pad_idx):
    """
    在資料集上測試模型效能

    此函數會在整個測試資料集上評估模型，計算各種效能指標。
    與訓練不同，這裡不會執行反向傳播或參數更新。

    參數：
        model: 要測試的 Transformer 模型
        test_loader: 測試資料載入器，提供批次化的測試資料
        device: 運算設備（'cpu' 或 'cuda'）
        pad_idx: 填充符號的索引值，計算指標時會忽略

    回傳：
        包含測試指標的字典（損失、準確率、困惑度等）
    """
    # 將模型設為評估模式
    # 這會停用 Dropout、Batch Normalization 的訓練行為
    model.eval()

    # 建立指標追蹤器，用於累積所有批次的統計資料
    metrics = TrainingMetrics()

    # 遍歷測試資料載入器中的所有批次
    # src: 來源序列（輸入）
    # tgt_input: 目標輸入序列（解碼器的輸入）
    # tgt_output: 目標輸出序列（用於計算損失和準確率）
    for src, tgt_input, tgt_output in test_loader:
        # ====== 資料準備階段 ======
        # 將所有資料移動到指定的運算設備
        src = src.to(device)
        tgt_input = tgt_input.to(device)
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
        # logits: 模型的原始輸出（未經 softmax）
        # 形狀為 (batch_size, tgt_seq_len, vocab_size)
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # ====== 損失計算 ======
        # 計算交叉熵損失（僅用於指標統計，不會用於訓練）
        # logits.reshape(-1, logits.size(-1)): 將 3D 張量重塑為 2D
        #   從 (batch_size, seq_len, vocab_size) 變成 (batch_size*seq_len, vocab_size)
        # tgt_output.reshape(-1): 將 2D 張量壓平為 1D
        #   從 (batch_size, seq_len) 變成 (batch_size*seq_len,)
        # ignore_index=pad_idx: 忽略填充符號，不計入損失
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # 預測輸出（展平）
            tgt_output.reshape(-1),               # 真實標籤（展平）
            ignore_index=pad_idx                  # 忽略填充符號
        )

        # ====== 更新指標 ======
        # 累積當前批次的統計資料到指標追蹤器
        # loss.item(): 將損失張量轉換為 Python 數值
        # logits: 用於計算預測準確率
        # tgt_output: 真實標籤，用於比較
        # pad_idx: 忽略填充符號
        metrics.update(loss.item(), logits, tgt_output, pad_idx)

    # 回傳整個測試集的平均指標
    return metrics.get_metrics()


@torch.no_grad()  # 裝飾器：停用梯度計算，節省記憶體
def interactive_test(model, vocab_size, device, start_token, end_token, task):
    """
    互動式測試 - 讓使用者輸入序列並生成輸出

    此函數提供一個互動式介面，讓使用者可以輸入自己的序列，
    然後看到模型的生成結果。這對於探索模型的能力非常有用。

    參數：
        model: 訓練好的 Transformer 模型
        vocab_size: 詞彙表大小，用於驗證輸入的有效性
        device: 運算設備（'cpu' 或 'cuda'）
        start_token: 開始符號的 ID
        end_token: 結束符號的 ID
        task: 任務類型（copy/reverse/sort），用於顯示預期輸出
    """
    # 將模型設為評估模式
    # 停用 Dropout 等訓練時的隨機性
    model.eval()

    # 印出互動模式的歡迎訊息和使用說明
    print("\n" + "="*60)
    print("🎮 互動模式")
    print("="*60)
    print(f"任務：{task}")
    print("輸入序列格式：空格分隔的數字（3-19）")
    print("範例：5 7 3 9")
    print("輸入 'quit' 離開")
    print("="*60 + "\n")

    # 主互動迴圈
    # 持續接受使用者輸入，直到使用者選擇離開
    while True:
        try:
            # ====== 獲取使用者輸入 ======
            # input(): 等待使用者輸入一行文字
            # strip(): 移除前後的空白字元
            user_input = input("輸入序列：").strip()

            # 檢查是否要離開
            # 支援多種退出指令：quit, exit, q（不區分大小寫）
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # ====== 解析輸入 ======
            # 將輸入字串分割並轉換為整數列表
            # split(): 以空格分割字串
            # int(x): 將每個元素轉換為整數
            # 列表推導式：[int(x) for x in user_input.split()]
            tokens = [int(x) for x in user_input.split()]

            # ====== 驗證輸入的有效性 ======
            # 檢查所有符號是否在有效範圍內
            # 有效範圍：3 到 vocab_size-1（0=PAD, 1=START, 2=END, 3-19=實際符號）
            # all(): 只有當所有元素都滿足條件時才回傳 True
            if not all(3 <= t < vocab_size for t in tokens):
                print(f"❌ 錯誤：所有符號必須在 3 到 {vocab_size-1} 之間")
                continue  # 跳過本次迴圈，繼續等待下一個輸入

            # ====== 建立輸入張量 ======
            # torch.tensor(): 將 Python 列表轉換為 PyTorch 張量
            # unsqueeze(0): 增加批次維度（從 1D 變成 2D）
            # to(device): 移動到指定的運算設備
            src = torch.tensor(tokens).unsqueeze(0).to(device)

            # ====== 生成輸出 ======
            # 使用模型的 generate 方法進行自回歸生成
            # 從 start_token 開始，逐步生成每個符號
            # 直到生成 end_token 或達到最大長度
            generated = model.generate(
                src,                    # 編碼器的輸入
                max_len=20,            # 最大生成長度
                start_token=start_token,  # 解碼器的起始符號
                end_token=end_token    # 生成結束的標記
            )

            # ====== 清理輸出 ======
            # 移除特殊符號（PAD, START, END）以便顯示
            # squeeze(0): 移除批次維度
            # cpu(): 將張量移回 CPU（如果在 GPU 上）
            # tolist(): 轉換為 Python 列表
            # 列表推導式：過濾掉特殊符號
            generated_clean = [x for x in generated.squeeze(0).cpu().tolist()
                             if x not in [0, start_token, end_token]]

            # ====== 顯示結果 ======
            # 將生成的符號列表轉換為字串並印出
            # map(str, generated_clean): 將每個數字轉換為字串
            # ' '.join(...): 用空格連接所有字串
            print(f"✨ 輸出：{' '.join(map(str, generated_clean))}")

            # ====== 顯示預期輸出（如果適用）======
            # 根據任務類型計算預期的正確輸出
            if task == 'copy':
                # 複製任務：輸出應該與輸入相同
                expected = tokens
            elif task == 'reverse':
                # 反轉任務：輸出應該是輸入的反轉
                # [::-1] 是 Python 的切片語法，表示反轉列表
                expected = tokens[::-1]
            elif task == 'sort':
                # 排序任務：輸出應該是輸入的排序結果
                expected = sorted(tokens)
            else:
                # 未知任務：無法計算預期輸出
                expected = None

            # 如果有預期輸出，則比較並顯示結果
            if expected:
                # 檢查生成結果是否與預期完全一致
                is_correct = generated_clean == expected
                # 根據正確性設定狀態訊息
                status = "✅ 正確！" if is_correct else "❌ 錯誤"
                # 印出狀態和預期輸出
                print(f"{status}（預期：{' '.join(map(str, expected))}）")

            # 印出空行，讓輸出更清晰
            print()

        except ValueError:
            # 處理輸入格式錯誤（例如輸入了非數字）
            print("❌ 無效的輸入。請使用空格分隔的數字。\n")
        except KeyboardInterrupt:
            # 處理 Ctrl+C 中斷
            break

    # 印出離開訊息
    print("\n👋 再見！")


def main():
    # ====== 建立命令列參數解析器 ======
    parser = argparse.ArgumentParser(description='測試 Transformer 模型')

    # checkpoint: 模型檢查點的路徑（必填參數）
    # required=True: 使用者必須提供此參數
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型檢查點的路徑')

    # task: 任務類型，用於顯示預期輸出
    # choices: 限制只能選擇這三種任務
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='任務類型')

    # num_samples: 測試樣本數量
    # 預設 1000 個，可以根據需要調整
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='測試樣本數量')

    # vocab_size: 詞彙表大小
    # 必須與訓練時使用的值一致
    parser.add_argument('--vocab-size', type=int, default=20,
                        help='詞彙表大小')

    # batch_size: 批次大小
    # 測試時的批次大小，只影響速度不影響結果
    parser.add_argument('--batch-size', type=int, default=64,
                        help='測試時的批次大小')

    # device: 運算設備
    # 'cpu' 或 'cuda'（GPU）
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='運算設備')

    # interactive: 啟用互動模式
    # action='store_true': 如果提供此參數，值為 True；否則為 False
    # 互動模式讓使用者可以輸入自己的序列進行測試
    parser.add_argument('--interactive', action='store_true',
                        help='啟用互動模式')

    # 解析命令列參數
    # 將使用者輸入的參數儲存在 args 物件中
    args = parser.parse_args()

    # ====== 設定運算設備 ======
    # 從參數中取得使用者指定的設備
    device = args.device

    # 檢查 CUDA（GPU）是否可用
    # 如果使用者指定 cuda 但系統沒有 GPU，則退回到 CPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，改用 CPU")
        device = 'cpu'

    # 印出測試標題
    print("\n" + "="*60)
    print("🧪 TRANSFORMER 模型測試")
    print("="*60 + "\n")

    # ====== 載入訓練配置 ======
    # 嘗試從檢查點目錄載入訓練時儲存的配置檔案
    # 這樣可以確保測試時使用與訓練時相同的模型架構

    # 取得檢查點檔案的父目錄
    # Path(args.checkpoint).parent: 從檔案路徑取得目錄路徑
    checkpoint_dir = Path(args.checkpoint).parent

    # 建立配置檔案的完整路徑
    # 通常是 checkpoints/config.json
    config_path = checkpoint_dir / 'config.json'

    # 檢查配置檔案是否存在
    if config_path.exists():
        # 如果配置檔案存在，則讀取它
        # 'with open()' 確保檔案會被正確關閉
        with open(config_path) as f:
            # 載入 JSON 檔案並解析為 Python 字典
            config = json.load(f)
        print("✅ 已載入訓練時的配置")
        # 提取模型相關的配置
        model_config = config['model']
    else:
        # 如果配置檔案不存在，使用預設值
        # 這些預設值應該與訓練時的預設值一致
        print("⚠️  找不到配置檔案，使用預設值")
        model_config = {
            'd_model': 128,         # 模型維度
            'num_heads': 4,        # 注意力頭數
            'num_layers': 2,       # 層數
            'd_ff': 512,          # 前饋網路維度
            'dropout': 0.1,       # Dropout 機率（測試時會自動停用）
            'vocab_size': args.vocab_size  # 詞彙表大小
        }

    # ====== 建立模型 ======
    # 使用與訓練時相同的架構建立模型
    model = create_transformer(
        src_vocab_size=model_config['vocab_size'],  # 來源詞彙表大小
        tgt_vocab_size=model_config['vocab_size'],  # 目標詞彙表大小
        d_model=model_config['d_model'],        # 模型維度
        num_heads=model_config['num_heads'],    # 注意力頭數
        num_layers=model_config['num_layers'],  # 編碼器和解碼器的層數
        d_ff=model_config['d_ff'],             # 前饋網路的隱藏層維度
        dropout=model_config['dropout']        # Dropout 機率（評估時會自動停用）
    )

    # 將模型移動到指定的運算設備
    # 確保模型和資料在同一個設備上
    model = model.to(device)

    # ====== 載入檢查點 ======
    # 從檢查點檔案載入訓練好的模型權重
    # checkpoint_info: 包含檢查點資訊的字典（如訓練週期、指標等）
    checkpoint_info = load_checkpoint(args.checkpoint, model)

    # 印出模型的參數數量
    # 使用千位分隔符讓數字更易讀
    print(f"模型參數數量：{model.count_parameters():,}")
    print()

    # ====== 在資料集上測試 ======
    print("📊 在資料集上測試...")

    # 建立測試資料載入器
    # 設定 train_split=0.0 和 val_split=0.0
    # 這樣所有資料都會被分配到測試集
    # _: 訓練載入器（不需要，用 _ 忽略）
    # _: 驗證載入器（不需要，用 _ 忽略）
    # test_loader: 測試資料載入器
    # dataset_info: 資料集資訊字典
    _, _, test_loader, dataset_info = create_dataloader(
        dataset_type=args.task,                 # 任務類型（copy/reverse/sort）
        batch_size=args.batch_size,             # 批次大小
        num_samples=args.num_samples,           # 測試樣本總數
        vocab_size=model_config['vocab_size'],  # 詞彙表大小
        train_split=0.0,  # 訓練集比例設為 0（不需要訓練資料）
        val_split=0.0     # 驗證集比例設為 0（所有資料都給測試集）
    )

    # 執行測試，計算各種效能指標
    # metrics: 包含損失、準確率、困惑度等的字典
    metrics = test_model(model, test_loader, device, dataset_info['pad_token'])

    # ====== 印出測試結果 ======
    print(f"✅ 測試結果：")
    # 損失值：數值越低越好
    print(f"   損失：{metrics['loss']:.4f}")
    # 符號準確率：單個符號的預測準確率（百分比）
    print(f"   符號準確率：{metrics['token_accuracy']:.2f}%")
    # 序列準確率：整個序列完全正確的比例（百分比）
    # 這是最重要的指標
    print(f"   序列準確率：{metrics['sequence_accuracy']:.2f}%")
    # 困惑度：越低表示模型越確定
    print(f"   困惑度：{metrics['perplexity']:.4f}")

    # ====== 互動模式 ======
    # 如果使用者啟用了互動模式
    if args.interactive:
        # 進入互動測試模式
        # 讓使用者可以輸入自己的序列並看到模型的生成結果
        interactive_test(
            model,                          # 訓練好的模型
            model_config['vocab_size'],     # 詞彙表大小
            device,                         # 運算設備
            dataset_info['start_token'],    # 開始符號
            dataset_info['end_token'],      # 結束符號
            args.task                       # 任務類型
        )


# 主程式入口點
# 當這個檔案直接執行時（而非被匯入時），會執行 main() 函數
if __name__ == '__main__':
    main()
