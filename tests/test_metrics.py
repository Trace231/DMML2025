import numpy as np

from segmentation_benchmark.metrics import MetricsAggregator, compute_metrics


def test_metrics_perfect_prediction():
    pred = np.ones((16, 16), dtype=np.int64)
    target = np.ones((16, 16), dtype=np.int64)
    metrics = compute_metrics(pred, target, num_classes=2)
    assert metrics.pixel_accuracy == 1.0
    assert metrics.mean_iou == 1.0
    assert metrics.mean_dice == 1.0


def test_metrics_aggregator_partial():
    aggregator = MetricsAggregator(num_classes=2)
    pred1 = np.zeros((4, 4), dtype=np.int64)
    target1 = np.zeros((4, 4), dtype=np.int64)
    aggregator.update(pred1, target1)
    pred2 = np.ones((4, 4), dtype=np.int64)
    target2 = np.zeros((4, 4), dtype=np.int64)
    aggregator.update(pred2, target2)
    summary = aggregator.summary()
    assert 0.5 <= summary.pixel_accuracy <= 1.0
    assert summary.mean_iou <= 1.0
