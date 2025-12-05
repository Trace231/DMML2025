#!/usr/bin/env python3
"""诊断 DataLoader 性能瓶颈的脚本。

使用方法:
    python scripts/diagnose_dataloader_perf.py --config configs/voc_benchmark.yaml
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.segmentation_benchmark.data import create_dataloaders


def benchmark_dataloader(loader: DataLoader, num_batches: int = 50, warmup: int = 5) -> dict:
    """测试 DataLoader 的性能。
    
    Returns:
        dict: 包含各种性能指标的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup
    print(f"Warming up with {warmup} batches...")
    for i, batch in enumerate(loader):
        if i >= warmup - 1:
            break
        # 模拟 GPU 传输
        if torch.cuda.is_available():
            batch["image"].to(device, non_blocking=True)
            batch["mask"].to(device, non_blocking=True)
    
    # 实际测试
    print(f"Benchmarking {num_batches} batches...")
    times = []
    data_times = []
    gpu_times = []
    
    start_total = time.time()
    loader_iter = iter(loader)
    
    for i in range(num_batches):
        # 测量数据加载时间
        data_start = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        data_time = time.time() - data_start
        data_times.append(data_time)
        
        # 测量 GPU 传输时间
        if torch.cuda.is_available():
            gpu_start = time.time()
            batch["image"].to(device, non_blocking=True)
            batch["mask"].to(device, non_blocking=True)
            torch.cuda.synchronize()  # 确保传输完成
            gpu_time = time.time() - gpu_start
            gpu_times.append(gpu_time)
        else:
            gpu_times.append(0.0)
        
        total_time = data_time + (gpu_times[-1] if torch.cuda.is_available() else 0)
        times.append(total_time)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_batches} batches...")
    
    total_time = time.time() - start_total
    
    # 计算统计信息
    stats = {
        "num_batches": num_batches,
        "total_time": total_time,
        "avg_time_per_batch": sum(times) / len(times),
        "avg_data_time": sum(data_times) / len(data_times),
        "avg_gpu_time": sum(gpu_times) / len(gpu_times) if gpu_times else 0,
        "min_data_time": min(data_times),
        "max_data_time": max(data_times),
        "median_data_time": sorted(data_times)[len(data_times) // 2],
        "throughput_batches_per_sec": num_batches / total_time,
        "data_loading_ratio": sum(data_times) / total_time,
        "gpu_transfer_ratio": sum(gpu_times) / total_time if gpu_times else 0,
    }
    
    return stats


def diagnose_bottleneck(stats: dict, num_workers: int, batch_size: int) -> list:
    """诊断性能瓶颈并给出建议。"""
    issues = []
    suggestions = []
    
    # 检查数据加载时间
    if stats["avg_data_time"] > 0.1:  # 如果每个 batch 加载时间超过 100ms
        issues.append("数据加载时间较长")
        if stats["data_loading_ratio"] > 0.5:
            issues.append("数据加载是主要瓶颈 (>50% 时间)")
    
    # 检查 worker 数量
    if num_workers > 16:
        issues.append(f"Worker 数量 ({num_workers}) 可能过多")
        suggestions.append(f"建议减少到 8-16 个 workers（经验法则：2-4 * CPU 核心数）")
    
    if num_workers == 0:
        issues.append("使用单线程数据加载")
        suggestions.append("建议设置 num_workers=4-8 以利用多核 CPU")
    
    # 检查 GPU 传输
    if stats["gpu_transfer_ratio"] > 0.3:
        issues.append("GPU 传输占用较多时间")
        suggestions.append("确保使用 pin_memory=True 和 non_blocking=True")
    
    # 检查吞吐量
    if stats["throughput_batches_per_sec"] < 1.0:
        issues.append("吞吐量较低 (<1 batch/s)")
        suggestions.append("考虑增加 batch_size 或优化数据预处理")
    
    # 检查数据加载时间波动
    if stats["max_data_time"] / stats["min_data_time"] > 3.0:
        issues.append("数据加载时间波动较大")
        suggestions.append("可能是 I/O 瓶颈或数据预处理复杂度不一致")
    
    return issues, suggestions


def main():
    parser = argparse.ArgumentParser(description="诊断 DataLoader 性能")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--num-batches", type=int, default=50, help="测试的 batch 数量")
    parser.add_argument("--warmup", type=int, default=5, help="预热 batch 数量")
    parser.add_argument("--test-workers", type=int, nargs="+", default=None,
                       help="要测试的 worker 数量列表（例如：0 4 8 16）")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_cfg = config.get("dataset", {})
    default_num_workers = dataset_cfg.get("num_workers", 0)
    
    # 确定要测试的 worker 数量
    if args.test_workers:
        test_workers = args.test_workers
    else:
        # 默认测试几个常见的值
        test_workers = [0, 4, 8, 16, default_num_workers]
        test_workers = sorted(set(test_workers))  # 去重并排序
    
    print("=" * 80)
    print("DataLoader 性能诊断工具")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"默认 num_workers: {default_num_workers}")
    print(f"测试的 worker 数量: {test_workers}")
    print(f"测试 batch 数量: {args.num_batches}")
    print("=" * 80)
    print()
    
    results = {}
    
    for num_workers in test_workers:
        print(f"\n{'='*80}")
        print(f"测试 num_workers = {num_workers}")
        print(f"{'='*80}")
        
        # 临时修改配置
        test_cfg = dataset_cfg.copy()
        test_cfg["num_workers"] = num_workers
        
        # 创建 DataLoader
        try:
            dataloaders = create_dataloaders(test_cfg)
            train_loader = dataloaders["train_loader"]
            
            # 检查 DataLoader 配置
            print(f"\nDataLoader 配置:")
            print(f"  num_workers: {train_loader.num_workers}")
            print(f"  batch_size: {train_loader.batch_size}")
            print(f"  pin_memory: {getattr(train_loader, 'pin_memory', False)}")
            print(f"  prefetch_factor: {getattr(train_loader, 'prefetch_factor', 2)}")
            print(f"  persistent_workers: {getattr(train_loader, 'persistent_workers', False)}")
            
            # 运行基准测试
            stats = benchmark_dataloader(train_loader, args.num_batches, args.warmup)
            results[num_workers] = stats
            
            # 打印结果
            print(f"\n性能统计:")
            print(f"  总时间: {stats['total_time']:.2f}s")
            print(f"  平均每 batch 时间: {stats['avg_time_per_batch']:.4f}s")
            print(f"  平均数据加载时间: {stats['avg_data_time']:.4f}s")
            if stats['avg_gpu_time'] > 0:
                print(f"  平均 GPU 传输时间: {stats['avg_gpu_time']:.4f}s")
            print(f"  吞吐量: {stats['throughput_batches_per_sec']:.2f} batches/s")
            print(f"  数据加载占比: {stats['data_loading_ratio']*100:.1f}%")
            if stats['gpu_transfer_ratio'] > 0:
                print(f"  GPU 传输占比: {stats['gpu_transfer_ratio']*100:.1f}%")
            print(f"  数据加载时间范围: {stats['min_data_time']:.4f}s - {stats['max_data_time']:.4f}s")
            
            # 诊断
            issues, suggestions = diagnose_bottleneck(
                stats, num_workers, train_loader.batch_size
            )
            
            if issues:
                print(f"\n⚠️  发现的问题:")
                for issue in issues:
                    print(f"  - {issue}")
            
            if suggestions:
                print(f"\n💡 建议:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
        
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 比较结果
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("性能比较")
        print(f"{'='*80}")
        print(f"{'num_workers':<12} {'吞吐量 (batches/s)':<20} {'平均加载时间 (s)':<20}")
        print("-" * 80)
        for num_workers in sorted(results.keys()):
            stats = results[num_workers]
            print(f"{num_workers:<12} {stats['throughput_batches_per_sec']:<20.2f} {stats['avg_data_time']:<20.4f}")
        
        # 找出最佳配置
        best_workers = max(results.keys(), key=lambda w: results[w]['throughput_batches_per_sec'])
        best_throughput = results[best_workers]['throughput_batches_per_sec']
        print(f"\n✅ 最佳配置: num_workers={best_workers} (吞吐量: {best_throughput:.2f} batches/s)")
        
        if best_workers != default_num_workers:
            print(f"⚠️  当前配置使用 num_workers={default_num_workers}，建议改为 {best_workers}")


if __name__ == "__main__":
    main()

