"""
綜合基準測試套件

驗證所有任務都能使用推薦參數成功訓練。
執行此程式可以驗證訓練流程是否正常運作。

使用方式：
    python benchmark.py                    # 執行所有任務
    python benchmark.py --task copy        # 執行特定任務
    python benchmark.py --quick            # 快速模式（較少訓練週期）
"""

# 引入命令列參數解析模組，處理使用者輸入的基準測試參數
import argparse
# 引入子程序模組，用於執行外部命令（訓練腳本）
import subprocess
# 引入 JSON 模組，用於儲存測試結果
import json
# 引入時間模組，用於計算測試執行時間
import time
# 引入路徑處理模組，用於處理檔案路徑
from pathlib import Path
# 引入日期時間模組，用於生成時間戳記
from datetime import datetime


class TaskBenchmark:
    """
    單一任務的基準測試配置

    此類別儲存一個任務的所有基準測試參數，
    包括訓練週期數、學習率、預期準確率等。

    屬性：
        name: 任務名稱（copy/reverse/sort）
        epochs: 訓練週期數
        fixed_lr: 固定學習率
        min_accuracy: 最低預期序列準確率（百分比）
        description: 任務描述
        result: 測試結果（初始為 None，執行後會填入）
    """

    def __init__(self, name, epochs, fixed_lr, min_accuracy, description):
        # 任務名稱（例如 'copy', 'reverse', 'sort'）
        self.name = name
        # 訓練週期數（遍歷訓練集的次數）
        self.epochs = epochs
        # 固定學習率（控制參數更新的步長）
        self.fixed_lr = fixed_lr
        # 最低預期序列準確率（用於判斷測試是否通過）
        self.min_accuracy = min_accuracy
        # 任務的簡短描述
        self.description = description
        # 測試結果（初始為 None，測試執行後會儲存結果字典）
        self.result = None


# ====== 標準基準測試配置 ======
# 每個任務的完整訓練配置，經過調整以達到良好效能
BENCHMARKS = {
    'copy': TaskBenchmark(
        name='copy',            # 任務名稱：複製
        epochs=20,              # 訓練週期數：20（複製是最簡單的任務）
        fixed_lr=0.001,         # 固定學習率：0.001
        min_accuracy=95.0,      # 最低預期準確率：95%（複製任務應該很容易達到高準確率）
        description='精確複製序列'  # 任務描述
    ),
    'reverse': TaskBenchmark(
        name='reverse',         # 任務名稱：反轉
        epochs=30,              # 訓練週期數：30（比複製難，需要更多訓練）
        fixed_lr=0.001,         # 固定學習率：0.001
        min_accuracy=80.0,      # 最低預期準確率：80%（反轉比複製難）
        description='反轉輸入序列'  # 任務描述
    ),
    'sort': TaskBenchmark(
        name='sort',            # 任務名稱：排序
        epochs=50,              # 訓練週期數：50（最困難的任務，需要最多訓練）
        fixed_lr=0.0005,        # 固定學習率：0.0005（比其他任務低，因為任務更複雜）
        min_accuracy=65.0,      # 最低預期準確率：65%（排序最難，預期準確率較低）
        description='將數字由小到大排序'  # 任務描述
    )
}

# ====== 快速模式配置（用於 CI/CD）======
# 訓練週期數較少的配置，用於快速驗證程式碼是否正常運作
# 適合在持續整合/持續部署（CI/CD）流程中使用
QUICK_BENCHMARKS = {
    'copy': TaskBenchmark(
        name='copy',            # 任務名稱：複製
        epochs=10,              # 訓練週期數：10（僅為標準模式的一半）
        fixed_lr=0.001,         # 固定學習率：0.001（與標準模式相同）
        min_accuracy=85.0,      # 最低預期準確率：85%（比標準模式低，因為訓練較少）
        description='複製序列（快速）'  # 任務描述
    ),
    'reverse': TaskBenchmark(
        name='reverse',         # 任務名稱：反轉
        epochs=15,              # 訓練週期數：15（僅為標準模式的一半）
        fixed_lr=0.001,         # 固定學習率：0.001
        min_accuracy=60.0,      # 最低預期準確率：60%（比標準模式低）
        description='反轉序列（快速）'  # 任務描述
    ),
    'sort': TaskBenchmark(
        name='sort',            # 任務名稱：排序
        epochs=25,              # 訓練週期數：25（僅為標準模式的一半）
        fixed_lr=0.0005,        # 固定學習率：0.0005（與標準模式相同）
        min_accuracy=45.0,      # 最低預期準確率：45%（比標準模式低）
        description='排序數字（快速）'  # 任務描述
    )
}


def run_training(benchmark: TaskBenchmark) -> dict:
    """
    執行單一任務的訓練

    此函數會使用指定的參數執行訓練腳本，
    並解析輸出以提取效能指標。

    參數：
        benchmark: TaskBenchmark 物件，包含任務的所有配置

    回傳：
        包含測試結果的字典（準確率、損失、執行時間等）
    """
    # 印出基準測試標題和配置資訊
    # '='*70: 建立 70 個等號的分隔線
    print(f"\n{'='*70}")
    # 印出任務名稱（全部大寫）
    print(f"基準測試：{benchmark.name.upper()}")
    print(f"{'='*70}")
    # 印出任務描述
    print(f"描述：{benchmark.description}")
    # 印出訓練週期數
    print(f"訓練週期數：{benchmark.epochs}")
    # 印出學習率
    print(f"學習率：{benchmark.fixed_lr}")
    # 印出目標準確率（必須達到的最低準確率）
    print(f"目標準確率：>={benchmark.min_accuracy}%")
    print(f"{'='*70}\n")

    # ====== 建立訓練命令 ======
    # 組合要執行的命令列指令
    # cmd 是一個列表，每個元素是指令的一部分
    cmd = [
        'python', 'train.py',                       # 執行 train.py 腳本
        '--task', benchmark.name,                   # 任務名稱
        '--epochs', str(benchmark.epochs),          # 訓練週期數（轉為字串）
        '--fixed-lr', str(benchmark.fixed_lr),      # 固定學習率（轉為字串）
        '--label-smoothing', '0.0',                 # 標籤平滑設為 0（基準測試不使用）
        '--dropout', '0.0',                         # Dropout 設為 0（基準測試不使用）
        '--checkpoint-dir', f'benchmarks/{benchmark.name}'  # 檢查點儲存目錄
    ]

    # ====== 執行訓練 ======
    # 記錄開始時間
    # time.time(): 取得目前的時間戳記（秒）
    start_time = time.time()

    try:
        # 執行子程序（訓練腳本）
        # subprocess.run(): 執行外部命令並等待完成
        result = subprocess.run(
            cmd,                    # 要執行的命令（列表形式）
            capture_output=True,    # 捕獲標準輸出和標準錯誤
            text=True,              # 以文字模式處理輸出（而非位元組）
            timeout=3600            # 超時限制：3600 秒（1 小時）
        )

        # 計算經過的時間
        # time.time() - start_time: 結束時間減去開始時間
        elapsed = time.time() - start_time

        # ====== 檢查訓練是否失敗 ======
        # returncode: 程序的退出代碼（0 表示成功，非 0 表示失敗）
        if result.returncode != 0:
            # 如果訓練崩潰（退出代碼非 0）
            print(f"[失敗] 訓練崩潰")
            # 印出錯誤訊息
            # result.stderr: 捕獲的標準錯誤輸出
            print(f"錯誤：{result.stderr}")
            # 回傳失敗結果字典
            return {
                'status': 'crashed',    # 狀態：崩潰
                'error': result.stderr, # 錯誤訊息
                'time': elapsed        # 執行時間
            }

        # ====== 解析輸出以提取最終指標 ======
        # 取得訓練腳本的標準輸出
        # result.stdout: 捕獲的標準輸出（包含所有印出的內容）
        output = result.stdout

        # 引入正規表達式模組（用於從文字中提取特定模式）
        import re

        # ====== 提取測試集指標（留存資料 - 最重要！）======
        # 從輸出中尋找測試集結果
        # 正規表達式說明：
        # r'Test Set Results:.*?Seq Acc:\s+([\d.]+)%'
        # - Test Set Results:: 尋找這個文字
        # - .*?: 匹配任意字元（非貪婪模式）
        # - Seq Acc:: 尋找序列準確率標籤
        # - \s+: 匹配一個或多個空白字元
        # - ([\d.]+): 捕獲一個或多個數字和小數點（準確率數值）
        # - %: 匹配百分號
        # re.DOTALL: 讓 . 也能匹配換行符號
        test_match = re.search(
            r'Test Set Results:.*?Seq Acc:\s+([\d.]+)%',
            output,
            re.DOTALL
        )

        # 如果找到測試集指標
        if test_match:
            # 提取測試準確率
            # test_match.group(1): 取得第一個捕獲群組（準確率數值）
            # float(): 轉換為浮點數
            test_accuracy = float(test_match.group(1))
            accuracy = test_accuracy
            # 記錄指標來源為測試集
            metric_source = "test"
        else:
            # 如果找不到測試集指標，退回到驗證準確率（向後相容）
            # 尋找最佳驗證準確率
            val_match = re.search(r'Best validation sequence accuracy: ([\d.]+)%', output)
            if val_match:
                # 提取驗證準確率
                accuracy = float(val_match.group(1))
                # 記錄指標來源為驗證集
                metric_source = "validation"
            else:
                # 如果兩者都找不到，印出警告
                print(f"  警告：無法從輸出中解析準確率")
                accuracy = 0.0
                metric_source = "unknown"

        # ====== 檢查是否通過測試 ======
        # 比較準確率是否達到最低要求
        passed = accuracy >= benchmark.min_accuracy
        # 根據結果設定狀態
        status = 'passed' if passed else 'failed'

        # ====== 印出結果 ======
        print(f"\n{'='*70}")
        if passed:
            # 如果通過測試
            print(f"[通過] {benchmark.name.upper()}")
        else:
            # 如果未通過測試
            print(f"[失敗] {benchmark.name.upper()}")
        print(f"{'='*70}")
        # 印出準確率和來源
        # metric_source.capitalize(): 將首字母大寫
        print(f"  {metric_source.capitalize()} 準確率：{accuracy:.2f}%（目標：>={benchmark.min_accuracy}%）")
        # 印出執行時間（秒和分鐘）
        print(f"  時間：{elapsed:.1f}秒（{elapsed/60:.1f}分鐘）")
        print(f"{'='*70}\n")

        # 回傳結果字典
        return {
            'status': status,           # 狀態（passed 或 failed）
            'accuracy': accuracy,       # 準確率
            'metric_source': metric_source,  # 指標來源（test 或 validation）
            'target': benchmark.min_accuracy,  # 目標準確率
            'passed': passed,           # 是否通過（布林值）
            'time': elapsed            # 執行時間（秒）
        }

    except subprocess.TimeoutExpired:
        # 處理超時異常
        # 如果訓練執行時間超過 timeout 參數（3600 秒 = 1 小時）
        # 計算已經過的時間
        elapsed = time.time() - start_time
        # 印出超時訊息
        print(f"[失敗] 訓練超時，經過 {elapsed:.1f}秒")
        # 回傳超時結果字典
        return {
            'status': 'timeout',  # 狀態：超時
            'time': elapsed      # 執行時間
        }

    except Exception as e:
        # 處理其他所有未預期的異常
        # Exception: Python 中所有異常的基礎類別
        # 計算已經過的時間
        elapsed = time.time() - start_time
        # 印出錯誤訊息
        # e: 異常物件
        print(f"[失敗] 未預期的錯誤：{e}")
        # 回傳錯誤結果字典
        return {
            'status': 'error',   # 狀態：錯誤
            'error': str(e),    # 錯誤訊息（轉為字串）
            'time': elapsed     # 執行時間
        }


def print_summary(results: dict):
    """
    印出最終摘要表格

    以表格形式顯示所有任務的測試結果，
    並計算總計時間和通過/失敗的數量。

    參數：
        results: 包含所有任務結果的字典
                 鍵是任務名稱，值是結果字典
    """
    # 印出摘要標題
    print("\n" + "="*70)
    print("基準測試摘要")
    print("="*70)

    # ====== 印出表格標題 ======
    # 使用格式化字串設定欄位寬度
    # :<15 表示靠左對齊，寬度 15 字元
    print(f"{'任務':<15} {'狀態':<10} {'準確率':<12} {'目標':<12} {'時間':<10}")
    # 印出分隔線
    print("-"*70)

    # 初始化統計計數器
    # total_time: 所有任務的總執行時間
    total_time = 0
    # passed_count: 通過測試的任務數量
    passed_count = 0
    # failed_count: 失敗測試的任務數量
    failed_count = 0

    # ====== 遍歷所有任務結果 ======
    # results.items(): 取得字典的所有鍵值對
    # task: 任務名稱（例如 'copy', 'reverse', 'sort'）
    # result: 該任務的結果字典
    for task, result in results.items():
        # 從結果字典中提取資訊
        # get(): 如果鍵不存在，回傳預設值
        # status: 測試狀態（passed, failed, crashed, timeout, error）
        status = result.get('status', 'unknown')
        # accuracy: 達到的準確率
        accuracy = result.get('accuracy', 0.0)
        # target: 目標準確率（最低要求）
        target = result.get('target', 0.0)
        # elapsed: 執行時間（秒）
        elapsed = result.get('time', 0.0)

        # 累加總執行時間
        total_time += elapsed

        # ====== 根據狀態設定狀態字串 ======
        if status == 'passed':
            # 通過測試
            status_str = '[通過] PASS'
            passed_count += 1  # 增加通過計數
        elif status == 'failed':
            # 未通過測試（準確率不足）
            status_str = '[失敗] FAIL'
            failed_count += 1  # 增加失敗計數
        elif status == 'crashed':
            # 訓練崩潰
            status_str = '[崩潰] CRASH'
            failed_count += 1  # 增加失敗計數
        elif status == 'timeout':
            # 訓練超時
            status_str = '[超時] TIMEOUT'
            failed_count += 1  # 增加失敗計數
        else:
            # 未知狀態
            status_str = '[?] UNKNOWN'
            failed_count += 1  # 增加失敗計數

        # ====== 格式化準確率字串 ======
        # 檢查結果中是否有準確率資料
        if 'accuracy' in result:
            # 如果有，格式化為百分比（保留 2 位小數）
            acc_str = f"{accuracy:.2f}%"
            # 格式化目標準確率（帶 >= 符號）
            target_str = f">={target:.2f}%"
        else:
            # 如果沒有準確率資料（例如崩潰或超時）
            acc_str = "N/A"  # 顯示 N/A（不適用）
            target_str = f">={target:.2f}%"

        # ====== 格式化時間字串 ======
        # 以秒為單位，保留 1 位小數
        time_str = f"{elapsed:.1f}s"

        # ====== 印出表格行 ======
        # 使用格式化字串確保欄位對齊
        print(f"{task:<15} {status_str:<10} {acc_str:<12} {target_str:<12} {time_str:<10}")

    # 印出表格底部分隔線
    print("="*70)

    # ====== 印出總計統計 ======
    # 顯示通過和失敗的任務數量
    print(f"總計：{passed_count} 個通過，{failed_count} 個失敗")
    # 顯示總執行時間（秒和分鐘）
    print(f"總時間：{total_time:.1f}秒（{total_time/60:.1f}分鐘）")
    print("="*70)

    # ====== 儲存結果到 JSON 檔案 ======
    # 建立輸出目錄路徑
    output_dir = Path('benchmark_results')
    # 建立目錄（如果已存在則不報錯）
    # exist_ok=True: 如果目錄已存在，不會拋出異常
    output_dir.mkdir(exist_ok=True)

    # 生成時間戳記
    # strftime(): 將日期時間格式化為字串
    # '%Y%m%d_%H%M%S': 格式為 年月日_時分秒（例如 20260407_143025）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 建立輸出檔案路徑
    # 檔名包含時間戳記，避免覆蓋舊結果
    output_file = output_dir / f'benchmark_{timestamp}.json'

    # 開啟檔案並寫入 JSON 資料
    # 'w': 寫入模式（會覆蓋已存在的檔案）
    with open(output_file, 'w') as f:
        # 將結果序列化為 JSON 並寫入檔案
        # indent=2: 使用 2 個空格縮排，讓 JSON 更易讀
        json.dump({
            'timestamp': timestamp,      # 時間戳記
            'passed': passed_count,      # 通過的任務數量
            'failed': failed_count,      # 失敗的任務數量
            'total_time': total_time,    # 總執行時間
            'results': results          # 所有任務的詳細結果
        }, f, indent=2)

    # 印出儲存訊息
    print(f"\n[*] 結果已儲存至：{output_file}")

    # ====== 回傳退出代碼 ======
    # 如果所有任務都通過，回傳 0（成功）
    # 如果有任務失敗，回傳 1（失敗）
    # 這個退出代碼可以被 CI/CD 系統使用
    return 0 if failed_count == 0 else 1


def main():
    # ====== 建立命令列參數解析器 ======
    parser = argparse.ArgumentParser(description='Transformer 訓練基準測試')

    # task: 指定要執行的特定任務（可選）
    # 如果不提供，會執行所有任務
    parser.add_argument('--task', type=str, choices=['copy', 'reverse', 'sort'],
                        help='只執行特定任務')

    # quick: 快速模式
    # action='store_true': 如果提供此參數，值為 True
    # 快速模式使用較少的訓練週期，適合 CI/CD 流程
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（較少訓練週期，用於 CI/CD）')

    # 解析命令列參數
    args = parser.parse_args()

    # ====== 選擇基準測試配置 ======
    # 根據是否使用快速模式選擇不同的配置集
    # 三元運算式：條件 ? 真值 : 假值
    benchmarks = QUICK_BENCHMARKS if args.quick else BENCHMARKS

    # ====== 過濾任務（如果有指定）======
    # 如果使用者指定了特定任務
    if args.task:
        # 只保留該任務的配置
        # {args.task: benchmarks[args.task]}: 建立只包含一個任務的新字典
        benchmarks = {args.task: benchmarks[args.task]}

    # ====== 印出基準測試資訊 ======
    print("\n" + "="*70)
    print("TRANSFORMER 訓練基準測試套件")
    print("="*70)
    # 印出模式（快速或完整）
    print(f"模式：{'快速' if args.quick else '完整'}")
    # 印出要執行的任務列表
    # ', '.join(): 用逗號和空格連接任務名稱
    print(f"任務：{', '.join(benchmarks.keys())}")
    print("="*70)

    # ====== 執行基準測試 ======
    # 建立空字典來儲存所有任務的結果
    results = {}

    # 遍歷所有要測試的任務
    # benchmarks.items(): 取得所有任務及其配置
    for task, benchmark in benchmarks.items():
        # 執行該任務的訓練
        # run_training(): 執行訓練並回傳結果字典
        result = run_training(benchmark)

        # 將結果儲存到 results 字典中
        # 鍵是任務名稱，值是結果字典
        results[task] = result

        # 在結果中加入目標準確率
        # 這樣摘要表格就能顯示目標值
        result['target'] = benchmark.min_accuracy

    # ====== 印出摘要並取得退出代碼 ======
    # print_summary(): 印出摘要表格並回傳退出代碼
    # exit_code: 0 表示所有測試通過，1 表示有測試失敗
    exit_code = print_summary(results)

    # 回傳退出代碼
    return exit_code


# 主程式入口點
# 當這個檔案直接執行時（而非被匯入時），會執行 main() 函數
# exit(): 使用 main() 的回傳值作為程式的退出代碼
if __name__ == '__main__':
    exit(main())
