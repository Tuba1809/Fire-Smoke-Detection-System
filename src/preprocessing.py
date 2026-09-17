import cv2

def resize_image(image, width=800):
    """
    Resize an image while maintaining its aspect ratio.
    """
    height, original_width = image.shape[:2]

    if original_width == width:
        return image

    ratio = width / original_width
    new_height = int(height * ratio)

    return cv2.resize(image, (width, new_height))

def apply_blur(image):
    """
    Apply Gaussian blur to reduce image noise.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    """
    Convert a BGR image to HSV color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)