import numpy as np

from src.detection import find_regions
from src.classification import (
    classify_fire,
    classify_smoke,
    determine_final_status
)


def test_find_regions():
    """
    Test whether a sufficiently large region
    is detected from a binary mask.
    """

    mask = np.zeros((200, 200), dtype=np.uint8)

    mask[50:120, 50:130] = 255

    regions = find_regions(
        mask,
        min_area=500
    )

    assert len(regions) == 1
    assert regions[0]["area"] >= 500


def test_fire_classification():
    """
    Test classification of a fire-like region.
    """

    features = {
        "area": 1500,
        "width": 50,
        "height": 60,
        "aspect_ratio": 0.83,
        "mean_hue": 15,
        "mean_saturation": 180,
        "mean_value": 200,
        "max_value": 240,
        "mean_brightness": 200
    }

    result = classify_fire(features)

    assert result == "FIRE"


def test_non_fire_classification():
    """
    Test classification of a non-fire region.
    """

    features = {
        "area": 1000,
        "width": 50,
        "height": 50,
        "aspect_ratio": 1.0,
        "mean_hue": 38,
        "mean_saturation": 100,
        "mean_value": 130,
        "max_value": 160
    }

    result = classify_fire(features)

    assert result == "NON-FIRE"


def test_smoke_classification():
    """
    Test classification of a smoke-like region.
    """

    hsv_image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    hsv_image[:, :, 0] = 0
    hsv_image[:, :, 1] = 30

    hsv_image[:50, :, 2] = 130
    hsv_image[50:, :, 2] = 170

    smoke_mask = np.ones(
        (100, 100),
        dtype=np.uint8
    ) * 255

    region = {
        "x": 0,
        "y": 0,
        "width": 100,
        "height": 100,
        "area": 5000
    }

    result = classify_smoke(
        region,
        hsv_image,
        smoke_mask
    )

    assert result == "SMOKE"


def test_final_status_fire():
    """
    Test final FIRE status.
    """

    result = determine_final_status(
        fire_count=2,
        smoke_count=0
    )

    assert result == "FIRE"


def test_final_status_smoke():
    """
    Test final SMOKE status.
    """

    result = determine_final_status(
        fire_count=0,
        smoke_count=5
    )

    assert result == "SMOKE"


def test_final_status_fire_and_smoke():
    """
    Test combined FIRE + SMOKE status.
    """

    result = determine_final_status(
        fire_count=2,
        smoke_count=3
    )

    assert result == "FIRE + SMOKE"


def test_final_status_normal():
    """
    Test NORMAL status.
    """

    result = determine_final_status(
        fire_count=0,
        smoke_count=0
    )

    assert result == "NORMAL"