from segmentation_benchmark.evaluation import list_segmenters


def test_registry_contains_known_models():
    names = set(list_segmenters())
    expected = {"classical_crf", "fcn_resnet50", "deeplabv3_resnet50", "segformer_b0", "random_walker", "hybrid_unet_transformer", "cnn_crf", "crf_wrapper"}
    missing = expected - names
    assert not missing, f"Missing expected segmenters: {missing}"
