import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class GeoPoint:
    """
    Represents a detected bright point with position, radius, and brightness.
    """
    x: int
    y: int
    r: float
    brightness: int

    @classmethod
    def from_keypoint(cls, kp, image):
        """
        Create a GeoPoint from a blob detector keypoint and image data.
        """
        return cls(
            x=int(round(kp.pt[0])),
            y=int(round(kp.pt[1])),
            r=kp.size / 2,
            brightness=image[int(kp.pt[1]), int(kp.pt[0])]
        )

def detect_stars(image_path):
    """
    Detect bright star-like points in a grayscale image.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
    if gray is None:
        raise ValueError(f"Could not load image: {image_path}")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur image to reduce noise

    # Set up parameters for SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100      # Minimum pixel intensity to start detecting blobs
    params.maxThreshold = 255      # Maximum pixel intensity to consider
    params.filterByArea = True     # Enable filtering by area (size of blobs)
    params.minArea = 10            # Minimum area of a blob to keep
    params.maxArea = 40            # Maximum area of a blob to keep
    params.filterByColor = True    # Enable filtering by brightness (color)
    params.blobColor = 255         # Detect bright blobs (value 255 = white)

    detector = cv2.SimpleBlobDetector_create(params)  # Create the blob detector
    keypoints = detector.detect(blurred)              # Detect keypoints in the image

    # Convert OpenCV keypoints to GeoPoint format
    return [GeoPoint.from_keypoint(kp, gray) for kp in keypoints]
