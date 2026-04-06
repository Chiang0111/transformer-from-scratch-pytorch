"""
Integration tests for training pipeline

These tests verify that training actually works end-to-end.
They're slower than unit tests but catch real-world issues.

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import subprocess
import tempfile
import shutil
from pathlib import Path


class TestTrainingSmoke:
    """Smoke tests - verify training doesn't crash"""

    def test_copy_task_trains_successfully(self):
        """Test that copy task trains for a few epochs without crashing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '3',
                '--num-samples', '1000',  # Small dataset for speed
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            # Should not crash
            assert result.returncode == 0, f"Training crashed: {result.stderr}"

            # Should create checkpoints
            checkpoint_dir = Path(tmpdir)
            assert (checkpoint_dir / 'config.json').exists()
            assert (checkpoint_dir / 'checkpoint_latest.pt').exists()

    def test_reverse_task_trains_successfully(self):
        """Test that reverse task trains for a few epochs without crashing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'reverse',
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

            assert result.returncode == 0, f"Training crashed: {result.stderr}"

    def test_sort_task_trains_successfully(self):
        """Test that sort task trains for a few epochs without crashing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'sort',
                '--epochs', '3',
                '--num-samples', '1000',
                '--fixed-lr', '0.0005',
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

            assert result.returncode == 0, f"Training crashed: {result.stderr}"


class TestTrainingQuality:
    """Quality tests - verify training actually learns"""

    def test_copy_task_achieves_minimum_accuracy(self):
        """Test that copy task reaches at least 80% accuracy in 10 epochs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '10',
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
                timeout=600  # 10 minutes
            )

            assert result.returncode == 0, f"Training crashed: {result.stderr}"

            # Parse output for accuracy
            import re
            matches = re.findall(
                r'Val Loss:\s+([\d.]+)\s+\|\s+Token Acc:\s+([\d.]+)%\s+\|\s+Seq Acc:\s+([\d.]+)%',
                result.stdout
            )

            assert len(matches) > 0, "Could not find validation metrics in output"

            # Get final accuracy
            _, _, final_acc = matches[-1]
            final_acc = float(final_acc)

            # Should reach at least 80% sequence accuracy
            assert final_acc >= 80.0, \
                f"Copy task only reached {final_acc:.1f}% accuracy (expected ≥80%)"

    @pytest.mark.slow
    def test_reverse_task_achieves_minimum_accuracy(self):
        """Test that reverse task reaches at least 60% accuracy in 15 epochs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'reverse',
                '--epochs', '15',
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
                timeout=900  # 15 minutes
            )

            assert result.returncode == 0

            import re
            matches = re.findall(
                r'Seq Acc:\s+([\d.]+)%',
                result.stdout
            )

            final_acc = float(matches[-1])
            assert final_acc >= 60.0, \
                f"Reverse task only reached {final_acc:.1f}% (expected ≥60%)"


class TestTrainingRobustness:
    """Robustness tests - verify training handles edge cases"""

    def test_training_with_very_small_dataset(self):
        """Test that training works with minimal data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'python', 'train.py',
                '--task', 'copy',
                '--epochs', '5',
                '--num-samples', '100',  # Very small
                '--fixed-lr', '0.001',
                '--label-smoothing', '0.0',
                '--dropout', '0.0',
                '--checkpoint-dir', tmpdir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            assert result.returncode == 0

    def test_training_resume_works(self):
        """Test that resuming from checkpoint works"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 3 epochs
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

            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
            assert result1.returncode == 0

            # Resume and train 2 more epochs
            checkpoint_path = Path(tmpdir) / 'checkpoint_latest.pt'
            assert checkpoint_path.exists()

            cmd2 = [
                'python', 'train.py',
                '--resume', str(checkpoint_path),
                '--epochs', '5',  # Total 5 epochs (will train 2 more)
                '--checkpoint-dir', tmpdir
            ]

            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
            assert result2.returncode == 0

            # Should contain "Resuming from epoch" in output
            assert 'Resuming from' in result2.stdout or 'resume' in result2.stdout.lower()


class TestOverfitCapability:
    """Test that model can overfit (architecture verification)"""

    def test_model_can_overfit_single_batch(self):
        """Run the overfit test to verify architecture"""
        result = subprocess.run(
            ['python', 'test_overfit.py'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Overfit test failed: {result.stderr}"

        # Should contain SUCCESS message
        assert 'SUCCESS' in result.stdout, \
            "Overfit test did not report success"

        # Should reach very low loss
        assert 'Loss=0.0000' in result.stdout or 'Loss=0.0001' in result.stdout, \
            "Overfit test did not reach near-zero loss"


# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow", default=False),
    reason="need --runslow option to run"
)
