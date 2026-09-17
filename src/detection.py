import cv2


def find_regions(mask, min_area=500, max_area=None):
    """
    Detect connected regions from a binary mask.

    Parameters:
        mask: Binary segmentation mask.
        min_area: Minimum contour area to keep.
        max_area: Optional maximum contour area.

    Returns:
        List of detected regions.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        bounding_box_area = width * height

        if bounding_box_area > 0:
            extent = area / bounding_box_area
        else:
            extent = 0.0

        regions.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area": area,
            "extent": extent
        })

    return regions