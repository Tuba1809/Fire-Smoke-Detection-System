import cv2
import numpy as np
def clean_mask(mask):
    """
    Remove small noise and close small gaps
    in the segmented fire mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    # Remove small noise
    opened_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
    # Close small gaps
    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    return cleaned_mask