"""
互動式 Transformer 示範

精美的命令列介面示範，展示訓練好的模型實際運作。
比 Jupyter 更好 - 乾淨、可版本控制、無額外依賴。

使用方式：
    python demo.py                           # 互動模式
    python demo.py --checkpoint path.pt      # 使用特定檢查點
    python demo.py --task copy               # 指定任務
    python demo.py --show-attention          # 視覺化注意力權重
"""

# 引入命令列參數解析模組，處理使用者輸入的示範參數
import argparse
# 引入 PyTorch 深度學習框架
import torch
# 引入路徑處理模組，用於處理檔案路徑
from pathlib import Path
# 從 transformer 模組引入建立模型的工廠函數
from transformer import create_transformer
# 從 utils 模組引入遮罩建立函數（雖然本檔案未使用，但保留以備不時之需）
from utils import create_padding_mask, create_target_mask
# 從 datasets 模組引入建立資料載入器的函數
from datasets import create_dataloader


def print_banner(text):
    """
    印出精美的橫幅標題

    用等號線條框起標題文字，讓輸出更美觀易讀。

    參數：
        text: 要顯示的標題文字
    """
    # 設定橫幅寬度為 70 個字元
    width = 70
    # 印出上方分隔線
    print("\n" + "=" * width)
    # 印出標題文字（左側留 2 個空格）
    print(f"  {text}")
    # 印出下方分隔線和額外的空行
    print("=" * width + "\n")


def print_example(num, input_seq, expected, predicted, correct):
    """
    印出單個範例的詳細資訊

    以美觀的格式顯示輸入、預期輸出、實際輸出，
    並使用顏色標示正確或錯誤。

    參數：
        num: 範例編號
        input_seq: 輸入序列
        expected: 預期輸出
        predicted: 模型預測的輸出
        correct: 是否正確（布林值）
    """
    # 根據正確性設定狀態文字
    status = "✅ 正確" if correct else "❌ 錯誤"

    # ANSI 顏色代碼
    # \033[92m: 綠色（正確）
    # \033[91m: 紅色（錯誤）
    color_code = "\033[92m" if correct else "\033[91m"

    # 重置顏色代碼（恢復預設顏色）
    reset_code = "\033[0m"

    # 印出範例編號和狀態（帶顏色）
    print(f"\n{color_code}範例 {num}：{status}{reset_code}")
    # 印出輸入序列
    print(f"  輸入：    {input_seq}")
    # 印出預期輸出
    print(f"  預期：    {expected}")
    # 印出模型預測的輸出
    print(f"  預測：    {predicted}")


def demo_checkpoint(checkpoint_path: str, task: str, num_examples: int = 10):
    """
    使用訓練好的檢查點執行示範

    載入訓練好的模型，在測試範例上展示模型的預測結果。
    這讓使用者可以直觀地看到模型的效能。

    參數：
        checkpoint_path: 檢查點檔案的路徑
        task: 任務類型（copy/reverse/sort）
        num_examples: 要展示的範例數量（預設 10 個）
    """
    # 印出示範標題
    # task.upper(): 將任務名稱轉為大寫
    print_banner(f"TRANSFORMER 示範 - {task.upper()} 任務")

    # ====== 載入檢查點 ======
    print("📂 載入檢查點...")
    # torch.load(): 載入 PyTorch 儲存的檔案
    # map_location='cpu': 即使檢查點在 GPU 上儲存，也載入到 CPU
    # 這確保在沒有 GPU 的環境也能執行
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # ====== 讀取訓練配置 ======
    # 嘗試從檢查點目錄讀取配置檔案
    # 這樣可以取得訓練時使用的模型架構參數

    # 建立配置檔案路徑
    # Path(checkpoint_path).parent: 取得檢查點檔案的父目錄
    # / 'config.json': 拼接檔案名稱
    config_path = Path(checkpoint_path).parent / 'config.json'

    # 檢查配置檔案是否存在
    if config_path.exists():
        # 引入 json 模組（局部引入，只在需要時才引入）
        import json
        # 讀取並解析 JSON 配置檔案
        with open(config_path) as f:
            config = json.load(f)
            # 印出任務類型
            print(f"   任務：{config['task']}")
            # 印出模型架構資訊
            # d_model: 模型維度
            # num_layers: 層數
            # num_heads: 注意力頭數
            print(f"   模型：d_model={config['model']['d_model']}, "
                  f"layers={config['model']['num_layers']}, "
                  f"heads={config['model']['num_heads']}")
    print()

    # ====== 建立資料載入器 ======
    print("📊 載入測試資料...")
    # 建立資料載入器來取得測試範例
    # _: 訓練載入器（不需要）
    # val_loader: 驗證載入器（用於示範）
    # _: 測試載入器（不需要）
    # dataset_info: 資料集資訊字典
    _, val_loader, _, dataset_info = create_dataloader(
        dataset_type=task,      # 任務類型
        batch_size=1,          # 批次大小設為 1（一次處理一個範例，適合示範）
        num_samples=1000,      # 總樣本數
        vocab_size=20          # 詞彙表大小
    )
    print()

    # ====== 建立模型 ======
    print("🤖 建立模型...")
    # 使用與訓練時相同的架構建立模型
    # 如果配置檔案存在，使用配置中的參數；否則使用預設值
    model = create_transformer(
        src_vocab_size=dataset_info['vocab_size'],  # 來源詞彙表大小
        tgt_vocab_size=dataset_info['vocab_size'],  # 目標詞彙表大小
        d_model=config['model']['d_model'] if config_path.exists() else 128,  # 模型維度
        num_heads=config['model']['num_heads'] if config_path.exists() else 4,  # 注意力頭數
        num_layers=config['model']['num_layers'] if config_path.exists() else 2,  # 層數
        d_ff=config['model']['d_ff'] if config_path.exists() else 512,  # 前饋網路維度
        dropout=0.0  # 示範時不使用 Dropout（評估模式會自動停用，這裡設為 0 更明確）
    )

    # ====== 載入模型權重 ======
    # 從檢查點中載入訓練好的模型權重
    # checkpoint['model_state_dict']: 儲存的模型參數字典
    # load_state_dict(): 將參數載入到模型中
    model.load_state_dict(checkpoint['model_state_dict'])

    # 將模型設為評估模式
    # 停用 Dropout 和 Batch Normalization 的訓練行為
    model.eval()

    # ====== 印出模型資訊 ======
    # 印出模型的參數數量（使用千位分隔符）
    print(f"   參數數量：{model.count_parameters():,}")

    # 印出檢查點對應的訓練週期
    # checkpoint['epoch']: 儲存時的週期編號（從 0 開始）
    # +1: 顯示時轉換為人類習慣的編號（從 1 開始）
    print(f"   檢查點週期：{checkpoint['epoch'] + 1}")

    # ====== 印出驗證指標（如果有）======
    # 從檢查點中提取驗證指標
    if 'metrics' in checkpoint:
        # 取得驗證指標字典
        val_metrics = checkpoint['metrics'].get('val', {})
        if val_metrics:
            # 印出驗證時的序列準確率
            # get('sequence_accuracy', 0): 如果沒有此鍵，回傳 0
            print(f"   驗證準確率：{val_metrics.get('sequence_accuracy', 0):.2f}%")
    print()

    # ====== 在範例上測試 ======
    # 印出預測結果的標題
    print_banner("預測結果")

    # 初始化計數器
    # correct_count: 正確預測的數量
    correct_count = 0
    # total_count: 總範例數量
    total_count = 0

    # 使用 torch.no_grad() 停用梯度計算
    # 因為這是推論階段，不需要計算梯度
    with torch.no_grad():
        # 遍歷驗證資料載入器
        # i: 範例索引（從 0 開始）
        # src: 來源序列（輸入）
        # _: 目標輸入（不需要）
        # tgt_output: 目標輸出（預期答案）
        for i, (src, _, tgt_output) in enumerate(val_loader):
            # 只處理指定數量的範例
            # 當達到 num_examples 時停止
            if i >= num_examples:
                break

            # ====== 生成預測 ======
            # 使用模型的 generate 方法進行自回歸生成
            generated = model.generate(
                src,                              # 編碼器的輸入序列
                max_len=20,                      # 最大生成長度
                start_token=dataset_info['start_token'],  # 解碼器的起始符號
                end_token=dataset_info['end_token']       # 生成結束的標記符號
            )

            # ====== 清理序列（移除特殊符號）======
            # 清理來源序列
            # src[0]: 取得批次中的第一個（也是唯一一個）樣本
            # tolist(): 轉換為 Python 列表
            # 列表推導式：過濾掉特殊符號（PAD=0, START, END）
            src_clean = [x for x in src[0].tolist()
                        if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]

            # 清理預期輸出
            expected_clean = [x for x in tgt_output[0].tolist()
                            if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]

            # 清理模型生成的輸出
            predicted_clean = [x for x in generated[0].tolist()
                             if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]

            # ====== 檢查正確性 ======
            # 比較預測結果與預期輸出是否完全一致
            correct = predicted_clean == expected_clean

            # 如果正確，增加正確計數
            if correct:
                correct_count += 1

            # 增加總計數
            total_count += 1

            # ====== 印出範例 ======
            # 使用自訂的格式化函數印出範例詳情
            # i + 1: 範例編號（從 1 開始而非 0）
            print_example(i + 1, src_clean, expected_clean, predicted_clean, correct)

    # ====== 摘要統計 ======
    # 計算準確率
    # 公式：正確數量 / 總數量 × 100%
    accuracy = 100.0 * correct_count / total_count

    # 印出摘要標題
    print_banner("摘要")

    # 印出正確數量與總數量
    # 格式：正確數/總數
    print(f"  正確：{correct_count}/{total_count}")

    # 印出準確率（保留 2 位小數）
    print(f"  準確率：{accuracy:.2f}%")

    # ====== 根據準確率顯示整體評價 ======
    # 使用顏色標示效能等級
    if accuracy >= 90:
        # 準確率 >= 90%：卓越表現（綠色）
        print("\n  ✨ \033[92m卓越表現！\033[0m")
    elif accuracy >= 70:
        # 準確率 >= 70%：良好表現（黃色）
        print("\n  👍 \033[93m良好表現！\033[0m")
    elif accuracy >= 50:
        # 準確率 >= 50%：需要更多訓練（黃色）
        print("\n  🤔 \033[93m需要更多訓練\033[0m")
    else:
        # 準確率 < 50%：表現不佳（紅色）
        print("\n  ⚠️  \033[91m表現不佳 - 請檢查訓練\033[0m")

    print()


def interactive_mode(checkpoint_path: str, task: str):
    """
    互動模式 - 使用者輸入序列

    讓使用者可以輸入自己的序列，然後看到模型的預測結果。
    這提供了一個互動式介面來探索模型的能力。

    參數：
        checkpoint_path: 檢查點檔案的路徑
        task: 任務類型（copy/reverse/sort）
    """
    # 印出互動模式標題
    # task.upper(): 將任務名稱轉為大寫
    print_banner(f"互動模式 - {task.upper()} 任務")

    # ====== 載入模型（與 demo_checkpoint 相同的流程）======
    # 載入檢查點檔案
    # map_location='cpu': 載入到 CPU，確保相容性
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 嘗試載入配置檔案
    # 建立配置檔案路徑
    config_path = Path(checkpoint_path).parent / 'config.json'

    # 如果配置檔案存在，讀取它
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
    else:
        # 如果配置檔案不存在，使用預設值
        config = {'model': {'d_model': 128, 'num_heads': 4, 'num_layers': 2, 'd_ff': 512}}

    # ====== 建立模型 ======
    # 使用與訓練時相同的架構建立模型
    model = create_transformer(
        src_vocab_size=20,                  # 來源詞彙表大小（固定為 20）
        tgt_vocab_size=20,                  # 目標詞彙表大小（固定為 20）
        d_model=config['model']['d_model'],      # 模型維度
        num_heads=config['model']['num_heads'],  # 注意力頭數
        num_layers=config['model']['num_layers'], # 層數
        d_ff=config['model']['d_ff'],           # 前饋網路維度
        dropout=0.0                         # 不使用 Dropout（評估模式）
    )

    # 載入訓練好的權重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 設為評估模式
    model.eval()

    # 印出載入成功訊息和使用說明
    print("🤖 模型已載入！")
    print("\n輸入序列格式：空格分隔的數字（3-19）。")
    print("特殊符號：0=PAD（填充），1=START（開始），2=END（結束）")
    print("輸入 'quit' 離開。\n")

    # 主互動迴圈
    # 持續接受使用者輸入直到退出
    while True:
        # ====== 獲取使用者輸入 ======
        try:
            # input(): 等待使用者輸入
            # strip(): 移除前後空白
            user_input = input("輸入序列：").strip()

            # 檢查是否要離開
            # 支援多種退出指令（不區分大小寫）
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再見！👋\n")
                break

            # ====== 解析並驗證輸入 ======
            try:
                # 將輸入字串分割並轉換為整數列表
                # split(): 以空格分割
                # int(x): 將每個元素轉為整數
                sequence = [int(x) for x in user_input.split()]

                # 驗證：檢查序列是否為空
                if not sequence:
                    print("⚠️  空序列，請重新輸入\n")
                    continue  # 跳過本次迴圈

                # 驗證：檢查所有數字是否在有效範圍內
                # any(): 只要有一個元素滿足條件就回傳 True
                # 有效範圍：3-19（0=PAD, 1=START, 2=END, 3-19=實際符號）
                if any(x < 3 or x > 19 for x in sequence):
                    print("⚠️  數字必須在 3-19 之間\n")
                    continue  # 跳過本次迴圈

                # ====== 建立輸入張量 ======
                # 將列表轉換為 PyTorch 張量
                # [sequence]: 增加批次維度（變成 2D 張量）
                src = torch.tensor([sequence])

                # ====== 生成預測 ======
                # 使用 torch.no_grad() 停用梯度計算
                with torch.no_grad():
                    # 使用模型的 generate 方法進行自回歸生成
                    generated = model.generate(
                        src,              # 編碼器的輸入
                        max_len=20,      # 最大生成長度
                        start_token=1,   # 開始符號（START）
                        end_token=2      # 結束符號（END）
                    )

                # ====== 清理輸出 ======
                # 移除特殊符號（PAD=0, START=1, END=2）
                # generated[0]: 取得批次中的第一個樣本
                # tolist(): 轉換為 Python 列表
                # 列表推導式：過濾掉特殊符號
                output = [x for x in generated[0].tolist() if x not in [0, 1, 2]]

                # ====== 顯示結果 ======
                print(f"輸出：{output}")

                # ====== 顯示預期輸出（如果適用）======
                # 根據任務類型計算預期的正確輸出
                if task == 'copy':
                    # 複製任務：輸出應該與輸入相同
                    expected = sequence
                elif task == 'reverse':
                    # 反轉任務：輸出應該是輸入的反轉
                    # list(reversed(sequence)): 反轉列表
                    expected = list(reversed(sequence))
                elif task == 'sort':
                    # 排序任務：輸出應該是輸入的排序結果
                    expected = sorted(sequence)
                else:
                    # 未知任務：無法計算預期輸出
                    expected = None

                # 如果有預期輸出，則比較並顯示
                if expected is not None:
                    # 檢查輸出是否與預期一致
                    correct = output == expected
                    # 根據正確性設定狀態訊息
                    status = "✅ 正確" if correct else "❌ 錯誤"
                    # 印出預期輸出
                    print(f"預期：{expected}")
                    # 印出狀態和空行
                    print(f"{status}\n")
                else:
                    # 如果沒有預期輸出，只印出空行
                    print()

            except ValueError:
                # 處理輸入格式錯誤（例如輸入了非數字）
                print("⚠️  無效的輸入。請使用空格分隔的整數。\n")

        except (KeyboardInterrupt, EOFError):
            # 處理 Ctrl+C 或 Ctrl+D 中斷
            print("\n\n再見！👋\n")
            break


def main():
    # ====== 建立命令列參數解析器 ======
    parser = argparse.ArgumentParser(description='Transformer 示範程式')

    # checkpoint: 檢查點檔案路徑（可選）
    # 如果不提供，會使用預設路徑 checkpoints/checkpoint_best.pt
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='檢查點路徑（預設：checkpoints/checkpoint_best.pt）')

    # task: 任務類型
    # 用於顯示預期輸出和驗證結果
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='任務類型（預設：copy）')

    # num_examples: 要展示的範例數量
    # 只在非互動模式下使用
    parser.add_argument('--num-examples', type=int, default=10,
                        help='要展示的範例數量（預設：10）')

    # interactive: 啟用互動模式
    # action='store_true': 如果提供此參數，值為 True
    # 互動模式讓使用者可以輸入自己的序列
    parser.add_argument('--interactive', action='store_true',
                        help='互動模式 - 輸入您自己的序列')

    # 解析命令列參數
    args = parser.parse_args()

    # ====== 尋找檢查點檔案 ======
    # 如果使用者沒有指定檢查點路徑
    if args.checkpoint is None:
        # 使用預設路徑
        # Path('checkpoints') / 'checkpoint_best.pt': 組合路徑
        default_path = Path('checkpoints') / 'checkpoint_best.pt'

        # 檢查預設路徑是否存在
        if default_path.exists():
            # 如果存在，使用它
            # str(): 轉換為字串路徑
            checkpoint_path = str(default_path)
        else:
            # 如果不存在，顯示錯誤訊息和使用說明
            print("❌ 找不到檢查點！")
            print("   請先訓練模型：python train.py --task copy --epochs 20")
            print("   或指定路徑：python demo.py --checkpoint path/to/checkpoint.pt")
            return 1  # 回傳錯誤代碼 1
    else:
        # 如果使用者有指定檢查點路徑，使用它
        checkpoint_path = args.checkpoint

    # 驗證檢查點檔案是否存在
    # Path(checkpoint_path).exists(): 檢查檔案是否存在
    if not Path(checkpoint_path).exists():
        # 如果檔案不存在，顯示錯誤訊息
        print(f"❌ 找不到檢查點：{checkpoint_path}")
        return 1  # 回傳錯誤代碼 1

    # ====== 執行示範 ======
    # 根據使用者的選擇執行不同的模式
    if args.interactive:
        # 如果啟用互動模式
        # 讓使用者可以輸入自己的序列並看到預測結果
        interactive_mode(checkpoint_path, args.task)
    else:
        # 如果是一般示範模式
        # 在預設的測試範例上展示模型的預測結果
        demo_checkpoint(checkpoint_path, args.task, args.num_examples)

    # 回傳成功代碼 0
    return 0


# 主程式入口點
# 當這個檔案直接執行時（而非被匯入時），會執行 main() 函數
# exit(): 使用 main() 的回傳值作為程式的退出代碼
if __name__ == '__main__':
    exit(main())
