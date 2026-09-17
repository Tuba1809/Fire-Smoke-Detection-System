import cv2
import numpy as np

def create_fire_mask(hsv_image):
    """
    Create a binary mask for pixels that have
    color characteristics commonly associated with fire.
    """
    lower_fire = np.array([0, 100, 100])
    upper_fire = np.array([40, 255, 255])

    fire_mask = cv2.inRange(
        hsv_image,
        lower_fire,
        upper_fire
    )
    return fire_mask

def create_smoke_mask(hsv_image):
    """
    Create a binary mask for pixels that have
    visual characteristics commonly associated with smoke.
    """
    lower_smoke = np.array([0, 0, 80])
    upper_smoke = np.array([180, 100, 230])
    smoke_mask = cv2.inRange(
        hsv_image,
        lower_smoke,
        upper_smoke
    )
    return smoke_mask