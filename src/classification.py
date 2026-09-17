import numpy as np


def extract_features(region, hsv_image, fire_mask):
    """
    Extract visual features from a detected fire region.
    """

    x = region["x"]
    y = region["y"]
    width = region["width"]
    height = region["height"]
    area = region["area"]

    aspect_ratio = (
        width / height
        if height > 0
        else 0
    )

    roi_hsv = hsv_image[
        y:y + height,
        x:x + width
    ]

    roi_mask = fire_mask[
        y:y + height,
        x:x + width
    ]

    fire_pixels = roi_hsv[
        roi_mask > 0
    ]

    if len(fire_pixels) > 0:

        hue = fire_pixels[:, 0]
        saturation = fire_pixels[:, 1]
        value = fire_pixels[:, 2]

        mean_hue = float(np.mean(hue))
        mean_saturation = float(np.mean(saturation))
        mean_value = float(np.mean(value))
        max_value = float(np.max(value))

        # Percentage of detected pixels having
        # strong fire-like HSV characteristics.
        strong_fire_pixels = (
            (hue <= 25)
            & (saturation >= 150)
            & (value >= 150)
        )

        fire_color_fraction = float(
            np.mean(strong_fire_pixels)
        )

        # Percentage of pixels that are very bright.
        bright_pixels = (
            value >= 180
        )

        bright_fraction = float(
            np.mean(bright_pixels)
        )

    else:

        mean_hue = 0.0
        mean_saturation = 0.0
        mean_value = 0.0
        max_value = 0.0
        fire_color_fraction = 0.0
        bright_fraction = 0.0

    return {
        "area": float(area),
        "width": width,
        "height": height,
        "aspect_ratio": float(aspect_ratio),
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "max_value": max_value,
        "fire_color_fraction": fire_color_fraction,
        "bright_fraction": bright_fraction
    }


def classify_fire(features):
    """
    Classify a detected region as FIRE or NON-FIRE.

    Multiple visual characteristics are considered:
    - hue
    - saturation
    - brightness
    - region area
    - proportion of strong fire-like pixels
    """

    hue = features["mean_hue"]
    saturation = features["mean_saturation"]
    value = features["mean_value"]
    max_value = features["max_value"]
    area = features["area"]

    fire_color_fraction = features.get(
        "fire_color_fraction",
        1.0
    )

    bright_fraction = features.get(
        "bright_fraction",
        1.0
    )

    color_condition = (
        0 <= hue <= 25
        and saturation >= 120
        and value >= 120
    )

    brightness_condition = (
        max_value >= 180
    )

    area_condition = (
        area >= 500
    )

    strong_fire_color_condition = (
        fire_color_fraction >= 0.25
    )

    bright_pixel_condition = (
        bright_fraction >= 0.15
    )

    if (
        color_condition
        and brightness_condition
        and area_condition
        and strong_fire_color_condition
        and bright_pixel_condition
    ):
        return "FIRE"

    return "NON-FIRE"


def classify_smoke(region, hsv_image, smoke_mask):
    """
    Classify a detected region as SMOKE or NON-SMOKE.
    """

    x = region["x"]
    y = region["y"]
    width = region["width"]
    height = region["height"]

    roi_hsv = hsv_image[
        y:y + height,
        x:x + width
    ]

    roi_mask = smoke_mask[
        y:y + height,
        x:x + width
    ]

    smoke_pixels = roi_hsv[
        roi_mask > 0
    ]

    if len(smoke_pixels) == 0:
        return "NON-SMOKE"

    mean_saturation = float(
        np.mean(smoke_pixels[:, 1])
    )

    mean_value = float(
        np.mean(smoke_pixels[:, 2])
    )

    value_std = float(
        np.std(smoke_pixels[:, 2])
    )

    area = region["area"]

    extent = region.get(
        "extent",
        0.0
    )

    low_saturation = (
        mean_saturation <= 60
    )

    moderate_brightness = (
        90 <= mean_value <= 210
    )

    sufficient_area = (
        area >= 1500
    )

    brightness_variation = (
        8 <= value_std <= 60
    )

    irregular_shape = (
        extent <= 0.85
    )

    if (
        low_saturation
        and moderate_brightness
        and sufficient_area
        and brightness_variation
        and irregular_shape
    ):
        return "SMOKE"

    return "NON-SMOKE"


def determine_final_status(
    fire_count,
    smoke_count
):
    """
    Determine the overall detection status.
    """

    if fire_count > 0 and smoke_count > 0:
        return "FIRE + SMOKE"

    if fire_count > 0:
        return "FIRE"

    if smoke_count > 0:
        return "SMOKE"

    return "NORMAL"