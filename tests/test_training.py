"""
訓練流程的整合測試

這些測試驗證訓練流程能夠端到端正常運作。
它們比單元測試慢，但能捕捉真實世界的問題。

測試類型說明：
1. 煙霧測試（Smoke Tests）：驗證訓練不會崩潰
2. 品質測試（Quality Tests）：驗證訓練確實能學習
3. 穩健性測試（Robustness Tests）：驗證邊界情況處理
4. 過擬合測試（Overfit Tests）：驗證架構能力

執行方式：
- 標準測試：pytest tests/test_training.py -v
- 包含慢速測試：pytest tests/test_training.py -v --runslow

注意：
- 這些測試會實際執行訓練腳本
- 需要較長時間（數分鐘）
- 使用臨時目錄存儲檢查點
"""

import pytest  # Python 測試框架
import torch  # PyTorch（主要用於導入檢查）
import subprocess  # 用於執行外部命令（訓練腳本）
import tempfile  # 用於建立臨時目錄
import shutil  # 用於檔案操作
from pathlib import Path  # 用於路徑處理


class TestTrainingSmoke:
    """
    煙霧測試 - 驗證訓練不會崩潰

    煙霧測試的目的：
    - 快速檢查基本功能是否正常
    - 不關注訓練品質，只關注是否能執行
    - 如果連煙霧測試都過不了，代表有嚴重問題

    命名由來：
    - 來自硬體測試：開機後冒煙 = 失敗
    - 軟體測試：執行後崩潰 = 失敗
    """

    def test_copy_task_trains_successfully(self):
        """
        測試複製任務是否能成功訓練

        複製任務（Copy Task）：
        - 最簡單的序列到序列任務
        - 目標序列 = 源序列
        - 用於驗證基本的訓練流程

        測試策略：
        - 只訓練 3 個 epoch（快速）
        - 使用小資料集（1000 個樣本）
        - 使用臨時目錄存儲檢查點
        - 驗證訓練不崩潰且產生檢查點
        """
        # === 使用臨時目錄 ===
        # tempfile.TemporaryDirectory() 建立臨時目錄
        # with 語句結束後自動刪除
        with tempfile.TemporaryDirectory() as tmpdir:
            # === 建立訓練命令 ===
            cmd = [
                'python', 'train.py',  # 執行訓練腳本
                '--task', 'copy',      # 使用複製任務
                '--epochs', '3',       # 只訓練 3 個 epoch（快速測試）
                '--num-samples', '1000',  # 小資料集（加快訓練）
                '--fixed-lr', '0.001',     # 固定學習率（簡化測試）
                '--label-smoothing', '0.0',  # 關閉標籤平滑
                '--dropout', '0.0',          # 關閉 dropout（減少隨機性）
                '--checkpoint-dir', tmpdir   # 檢查點儲存到臨時目錄
            ]

            # === 執行訓練 ===
            # subprocess.run() 執行外部命令
            result = subprocess.run(
                cmd,
                capture_output=True,  # 捕獲標準輸出和標準錯誤
                text=True,            # 將輸出解碼為字串
                timeout=120           # 2 分鐘超時（防止卡住）
            )

            # === 驗證訓練成功 ===
            # returncode == 0 表示成功
            assert result.returncode == 0, \
                f"訓練崩潰了：{result.stderr}"

            # === 驗證檢查點檔案存在 ===
            checkpoint_dir = Path(tmpdir)

            # config.json：訓練配置
            assert (checkpoint_dir / 'config.json').exists(), \
                "缺少 config.json 檔案"

            # checkpoint_latest.pt：最新的模型檢查點
            assert (checkpoint_dir / 'checkpoint_latest.pt').exists(), \
                "缺少 checkpoint_latest.pt 檔案"

    def test_reverse_task_trains_successfully(self):
        """
        測試反轉任務是否能成功訓練

        反轉任務（Reverse Task）：
        - 目標序列是源序列的反轉
        - 例如：[1, 2, 3] → [3, 2, 1]
        - 比複製任務稍難，但仍然簡單
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'reverse',   # 使用反轉任務
                '--epochs', '3',
                '--num-samples', '1000',
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # 驗證訓練成功
            assert result.returncode == 0, \
                f"反轉任務訓練崩潰：{result.stderr}"

    def test_sort_task_trains_successfully(self):
        """
        測試排序任務是否能成功訓練

        排序任務（Sort Task）：
        - 將輸入序列排序
        - 例如：[3, 1, 2] → [1, 2, 3]
        - 比前兩個任務更難，需要全域理解

        注意：
        - 使用較小的學習率（0.0005 vs 0.001）
        - 排序任務對學習率更敏感
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'sort',      # 使用排序任務
                '--epochs', '3',
                '--num-samples', '1000',
                '--fixed-lr', '0.0005',  # 較小的學習率
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # 驗證訓練成功
            assert result.returncode == 0, \
                f"排序任務訓練崩潰：{result.stderr}"


class TestTrainingQuality:
    """
    品質測試 - 驗證訓練確實能學習

    煙霧測試只檢查不崩潰，品質測試檢查是否真的學到東西。

    驗證策略：
    - 訓練較多 epoch（讓模型有時間學習）
    - 檢查最終準確率是否達到預期閾值
    - 如果準確率太低，代表訓練有問題
    """

    def test_copy_task_achieves_minimum_accuracy(self):
        """
        測試複製任務是否達到最低準確率

        預期：
        - 複製任務很簡單
        - 10 個 epoch 後應該達到至少 80% 序列準確率
        - 如果達不到，可能是模型、訓練或資料有問題
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '10',      # 較多 epoch
                '--num-samples', '5000',  # 較多資料
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 分鐘超時
            )

            # 驗證訓練成功
            assert result.returncode == 0, \
                f"訓練崩潰：{result.stderr}"

            # === 解析輸出中的準確率 ===
            import re

            # 使用正則表達式從輸出中提取準確率
            # 尋找格式：Val Loss: X.XX | Token Acc: XX.XX% | Seq Acc: XX.XX%
            matches = re.findall(
                r'Val Loss:\s+([\d.]+)\s+\|\s+Token Acc:\s+([\d.]+)%\s+\|\s+Seq Acc:\s+([\d.]+)%',
                result.stdout
            )

            assert len(matches) > 0, \
                "無法在輸出中找到驗證指標"

            # === 提取最終準確率 ===
            # matches[-1] 是最後一個 epoch 的結果
            # [2] 是序列準確率
            _, _, final_acc = matches[-1]
            final_acc = float(final_acc)

            # === 驗證準確率達標 ===
            assert final_acc >= 80.0, \
                f"複製任務只達到 {final_acc:.1f}% 準確率（期望 ≥80%）"

    @pytest.mark.slow
    def test_reverse_task_achieves_minimum_accuracy(self):
        """
        測試反轉任務是否達到最低準確率

        預期：
        - 反轉任務比複製任務難
        - 15 個 epoch 後應該達到至少 60% 序列準確率

        標記為 slow：
        - 這個測試需要較長時間（15 分鐘）
        - 只有使用 --runslow 選項時才執行
        - pytest tests/ -v --runslow
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'reverse',
                '--epochs', '15',      # 更多 epoch
                '--num-samples', '5000',
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 分鐘超時
            )

            assert result.returncode == 0

            # 解析準確率
            import re
            matches = re.findall(
                r'Seq Acc:\s+([\d.]+)%',
                result.stdout
            )

            final_acc = float(matches[-1])

            # 反轉任務預期至少 60% 準確率
            assert final_acc >= 60.0, \
                f"反轉任務只達到 {final_acc:.1f}%（期望 ≥60%）"


class TestTrainingRobustness:
    """
    穩健性測試 - 驗證訓練處理邊界情況的能力

    測試各種極端或特殊情況：
    - 極小資料集
    - 從檢查點恢復訓練
    - 不同的超參數組合
    """

    def test_training_with_very_small_dataset(self):
        """
        測試使用極小資料集訓練

        目的：
        - 確保訓練在資料不足時不會崩潰
        - 驗證批次處理邏輯對小資料集仍然有效

        注意：
        - 不期望高準確率（資料太少）
        - 只要不崩潰就算通過
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '5',
                '--num-samples', '100',  # 極小資料集（只有 100 個樣本）
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 分鐘應該夠了
            )

            # 只要不崩潰就算通過
            assert result.returncode == 0, \
                f"小資料集訓練崩潰：{result.stderr}"

    def test_training_resume_works(self):
        """
        測試從檢查點恢復訓練是否正常運作

        訓練恢復的重要性：
        - 長時間訓練可能中斷（斷電、超時等）
        - 必須能夠從檢查點繼續，而不是重新開始
        - 節省計算資源和時間

        測試流程：
        1. 訓練 3 個 epoch 並儲存檢查點
        2. 從檢查點恢復並訓練到第 5 個 epoch
        3. 驗證恢復成功
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # === 第一階段：訓練 3 個 epoch ===
            cmd1 = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '3',
                '--num-samples', '500',
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result1 = subprocess.run(
                cmd1,
                capture_output=True,
                text=True,
                timeout=120
            )

            # 驗證第一階段成功
            assert result1.returncode == 0, \
                f"初始訓練失敗：{result1.stderr}"

            # === 驗證檢查點存在 ===
            checkpoint_path = Path(tmpdir) / 'checkpoint_latest.pt'
            assert checkpoint_path.exists(), \
                "檢查點檔案不存在"

            # === 第二階段：從檢查點恢復並繼續訓練 ===
            cmd2 = [
                'python', 'train.py',
                '--resume', str(checkpoint_path),  # 從檢查點恢復
                '--epochs', '5',  # 總共訓練到 5 個 epoch（會訓練 2 個額外 epoch）
                '--checkpoint-dir', tmpdir
            ]

            result2 = subprocess.run(
                cmd2,
                capture_output=True,
                text=True,
                timeout=120
            )

            # 驗證恢復訓練成功
            assert result2.returncode == 0, \
                f"恢復訓練失敗：{result2.stderr}"

            # === 驗證輸出中包含恢復資訊 ===
            # 訓練腳本應該輸出類似 "Resuming from epoch 3" 的訊息
            assert 'Resuming from' in result2.stdout or 'resume' in result2.stdout.lower(), \
                "輸出中沒有恢復訓練的提示"


class TestOverfitCapability:
    """
    過擬合能力測試 - 驗證模型架構正確

    過擬合測試的邏輯：
    - 如果模型連單一批次都無法過擬合，代表架構有問題
    - 健康的模型應該能夠記住（過擬合）小量資料
    - 這不代表我們希望過擬合，而是驗證模型「有能力」學習

    類比：
    - 就像測試汽車能否達到最高速度
    - 不是要飆車，而是確保引擎正常運作
    """

    def test_model_can_overfit_single_batch(self):
        """
        執行過擬合測試以驗證架構

        測試流程：
        - 執行 test_overfit.py 腳本
        - 該腳本訓練模型在單一批次上達到極低損失
        - 如果成功，證明架構沒有根本問題

        預期結果：
        - 損失應該降到接近 0（0.0000 或 0.0001）
        - 輸出應該包含 "SUCCESS" 訊息
        """
        # === 執行過擬合測試腳本 ===
        result = subprocess.run(
            ['python', 'test_overfit.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 分鐘應該夠了
        )

        # === 驗證測試成功 ===
        assert result.returncode == 0, \
            f"過擬合測試失敗：{result.stderr}"

        # === 驗證包含成功訊息 ===
        assert 'SUCCESS' in result.stdout, \
            "過擬合測試沒有報告成功"

        # === 驗證達到極低損失 ===
        # 輸出應該包含類似 "Loss=0.0000" 或 "Loss=0.0001"
        assert 'Loss=0.0000' in result.stdout or 'Loss=0.0001' in result.stdout, \
            "過擬合測試沒有達到接近零的損失"


# === 標記慢速測試 ===
# 這個程式碼片段定義了 @pytest.mark.slow 裝飾器的行為
# 只有在命令列包含 --runslow 時才執行標記為 slow 的測試
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow", default=False),
    reason="需要 --runslow 選項才執行慢速測試"
)
