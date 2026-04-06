"""
Comprehensive Benchmark Suite

Validates that all tasks train successfully with recommended parameters.
Run this to verify the training pipeline works correctly.

Usage:
    python benchmark.py                    # Run all tasks
    python benchmark.py --task copy        # Run specific task
    python benchmark.py --quick            # Quick mode (fewer epochs)
"""

import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


class TaskBenchmark:
    """Benchmark configuration for a single task"""

    def __init__(self, name, epochs, fixed_lr, min_accuracy, description):
        self.name = name
        self.epochs = epochs
        self.fixed_lr = fixed_lr
        self.min_accuracy = min_accuracy  # Minimum expected sequence accuracy
        self.description = description
        self.result = None


# Standard benchmark configurations
BENCHMARKS = {
    'copy': TaskBenchmark(
        name='copy',
        epochs=20,
        fixed_lr=0.001,
        min_accuracy=95.0,
        description='Copy sequences exactly'
    ),
    'reverse': TaskBenchmark(
        name='reverse',
        epochs=30,
        fixed_lr=0.001,
        min_accuracy=80.0,
        description='Reverse input sequences'
    ),
    'sort': TaskBenchmark(
        name='sort',
        epochs=50,
        fixed_lr=0.0005,
        min_accuracy=65.0,
        description='Sort numbers in ascending order'
    )
}

# Quick mode (for CI/CD)
QUICK_BENCHMARKS = {
    'copy': TaskBenchmark(
        name='copy',
        epochs=10,
        fixed_lr=0.001,
        min_accuracy=85.0,
        description='Copy sequences (quick)'
    ),
    'reverse': TaskBenchmark(
        name='reverse',
        epochs=15,
        fixed_lr=0.001,
        min_accuracy=60.0,
        description='Reverse sequences (quick)'
    ),
    'sort': TaskBenchmark(
        name='sort',
        epochs=25,
        fixed_lr=0.0005,
        min_accuracy=45.0,
        description='Sort numbers (quick)'
    )
}


def run_training(benchmark: TaskBenchmark) -> dict:
    """
    Run training for a single task

    Returns:
        dict with results (accuracy, loss, time, etc.)
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark.name.upper()}")
    print(f"{'='*70}")
    print(f"Description: {benchmark.description}")
    print(f"Epochs: {benchmark.epochs}")
    print(f"Learning Rate: {benchmark.fixed_lr}")
    print(f"Target Accuracy: >={benchmark.min_accuracy}%")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        'python', 'train.py',
        '--task', benchmark.name,
        '--epochs', str(benchmark.epochs),
        '--fixed-lr', str(benchmark.fixed_lr),
        '--label-smoothing', '0.0',
        '--dropout', '0.0',
        '--checkpoint-dir', f'benchmarks/{benchmark.name}'
    ]

    # Run training
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"[FAIL] FAILED: Training crashed")
            print(f"Error: {result.stderr}")
            return {
                'status': 'crashed',
                'error': result.stderr,
                'time': elapsed
            }

        # Parse output to extract final metrics
        output = result.stdout
        import re

        # Extract test set metrics (held-out data - most important!)
        # Look for: "Seq Acc:   XX.XX%" in Test Set Results section
        test_match = re.search(
            r'Test Set Results:.*?Seq Acc:\s+([\d.]+)%',
            output,
            re.DOTALL
        )

        if test_match:
            # Found test metrics - use these (proper held-out evaluation)
            test_accuracy = float(test_match.group(1))
            accuracy = test_accuracy
            metric_source = "test"
        else:
            # Fallback: use validation accuracy (for backwards compatibility)
            val_match = re.search(r'Best validation sequence accuracy: ([\d.]+)%', output)
            if val_match:
                accuracy = float(val_match.group(1))
                metric_source = "validation"
            else:
                print(f"  WARNING: Could not parse accuracy from output")
                accuracy = 0.0
                metric_source = "unknown"

        # Check if passed
        passed = accuracy >= benchmark.min_accuracy
        status = 'passed' if passed else 'failed'

        # Print result
        print(f"\n{'='*70}")
        if passed:
            print(f"[OK] PASSED: {benchmark.name.upper()}")
        else:
            print(f"[FAIL] FAILED: {benchmark.name.upper()}")
        print(f"{'='*70}")
        print(f"  {metric_source.capitalize()} Accuracy: {accuracy:.2f}% (target: >={benchmark.min_accuracy}%)")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"{'='*70}\n")

        return {
            'status': status,
            'accuracy': accuracy,
            'metric_source': metric_source,
            'target': benchmark.min_accuracy,
            'passed': passed,
            'time': elapsed
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[FAIL] FAILED: Training timed out after {elapsed:.1f}s")
        return {
            'status': 'timeout',
            'time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FAIL] FAILED: Unexpected error: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'time': elapsed
        }


def print_summary(results: dict):
    """Print final summary table"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    # Table header
    print(f"{'Task':<15} {'Status':<10} {'Accuracy':<12} {'Target':<12} {'Time':<10}")
    print("-"*70)

    total_time = 0
    passed_count = 0
    failed_count = 0

    for task, result in results.items():
        status = result.get('status', 'unknown')
        accuracy = result.get('accuracy', 0.0)
        target = result.get('target', 0.0)
        elapsed = result.get('time', 0.0)

        total_time += elapsed

        # Status emoji
        if status == 'passed':
            status_str = '[OK] PASS'
            passed_count += 1
        elif status == 'failed':
            status_str = '[FAIL] FAIL'
            failed_count += 1
        elif status == 'crashed':
            status_str = '[CRASH] CRASH'
            failed_count += 1
        elif status == 'timeout':
            status_str = '[TIMEOUT]  TIMEOUT'
            failed_count += 1
        else:
            status_str = '[?] UNKNOWN'
            failed_count += 1

        # Format accuracy
        if 'accuracy' in result:
            acc_str = f"{accuracy:.2f}%"
            target_str = f">={target:.2f}%"
        else:
            acc_str = "N/A"
            target_str = f">={target:.2f}%"

        # Format time
        time_str = f"{elapsed:.1f}s"

        print(f"{task:<15} {status_str:<10} {acc_str:<12} {target_str:<12} {time_str:<10}")

    print("="*70)
    print(f"Total: {passed_count} passed, {failed_count} failed")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*70)

    # Save results to JSON
    output_dir = Path('benchmark_results')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmark_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'passed': passed_count,
            'failed': failed_count,
            'total_time': total_time,
            'results': results
        }, f, indent=2)

    print(f"\n[*] Results saved to: {output_file}")

    # Return exit code
    return 0 if failed_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer training')
    parser.add_argument('--task', type=str, choices=['copy', 'reverse', 'sort'],
                        help='Run specific task only')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer epochs, for CI/CD)')
    args = parser.parse_args()

    # Select benchmarks
    benchmarks = QUICK_BENCHMARKS if args.quick else BENCHMARKS

    # Filter by task if specified
    if args.task:
        benchmarks = {args.task: benchmarks[args.task]}

    print("\n" + "="*70)
    print("TRANSFORMER TRAINING BENCHMARK SUITE")
    print("="*70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Tasks: {', '.join(benchmarks.keys())}")
    print("="*70)

    # Run benchmarks
    results = {}

    for task, benchmark in benchmarks.items():
        result = run_training(benchmark)
        results[task] = result
        result['target'] = benchmark.min_accuracy

    # Print summary
    exit_code = print_summary(results)

    return exit_code


if __name__ == '__main__':
    exit(main())
