import cv2
import numpy as np
import os
import itertools
from collections import defaultdict

RESULTS_DIR = "results"

def ensure_results_dir():
    """
    Ensure the output directory exists.
    Creates the 'results' directory if it doesn't already exist.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def read_grayscale(path):
    """
    Load an image in grayscale format.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

def read_color(path):
    """
    Load an image in color format.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Color image.
    """
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

def detect_blobs(gray_img):
    """
    Detect bright circular blobs in a grayscale image using OpenCV's blob detector.

    Args:
        gray_img (np.ndarray): Grayscale input image.

    Returns:
        list of cv2.KeyPoint: Detected blobs.
    """
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 40
    params.filterByColor = True
    params.blobColor = 255  # Detect bright spots

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(blur)

def extract_star_data(keypoints, img):
    """
    Convert detected keypoints to structured star data.

    Args:
        keypoints (list of cv2.KeyPoint): Detected keypoints.
        img (np.ndarray): Original grayscale image for brightness info.

    Returns:
        list of dict: Each dict contains x, y, radius, and brightness of a star.
    """
    return [{
        "x": int(round(kp.pt[0])),
        "y": int(round(kp.pt[1])),
        "r": kp.size / 2,
        "b": img[int(round(kp.pt[1])), int(round(kp.pt[0]))]
    } for kp in keypoints]

def detect_stars(image_path):
    """
    Full star detection pipeline for an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        list of dict: Detected stars.
    """
    gray_img = read_grayscale(image_path)
    keypoints = detect_blobs(gray_img)
    return extract_star_data(keypoints, gray_img)

def draw_stars(image, stars, annotate=True):
    """
    Draw circles and optional coordinate labels around detected stars.

    Args:
        image (np.ndarray): Color image to draw on.
        stars (list of dict): Detected star data.
        annotate (bool): Whether to draw coordinates near stars.

    Returns:
        np.ndarray: Image with annotations.
    """
    for s in stars:
        center = (s["x"], s["y"])
        cv2.circle(image, center, int(round(s["r"])), (0, 255, 0), 1)
        if annotate:
            cv2.putText(image, f"{s['x']},{s['y']}", (s["x"] + 4, s["y"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return image

def save_star_image(image_path, stars, filename):
    """
    Save an image with stars annotated.

    Args:
        image_path (str): Original image path.
        stars (list of dict): Star data.
        filename (str): Filename to save in results folder.
    """
    img = read_color(image_path)
    img_with_stars = draw_stars(img, stars)
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, img_with_stars)

def quantize(value, step):
    """
    Quantize a continuous value to the nearest step.

    Args:
        value (float): The input value.
        step (float): Quantization resolution.

    Returns:
        float: Quantized value.
    """
    return round(value / step) * step

def geometric_hash(stars, q=0.05):
    """
    Build a geometric hash table based on relative star positions.

    Args:
        stars (list of dict): Star coordinates.
        q (float): Quantization step.

    Returns:
        dict: Hash table mapping quantized positions to triangle triplets.
    """
    table = defaultdict(list)
    coords = np.array([(s["x"], s["y"]) for s in stars])

    for i, j in itertools.combinations(range(len(coords)), 2):
        p1, p2 = coords[i], coords[j]
        vec = p2 - p1
        d = np.linalg.norm(vec)
        if d < 10:
            continue
        u = vec / d
        v = np.array([-u[1], u[0]])  # perpendicular

        for k, p in enumerate(coords):
            if k in (i, j):
                continue
            rel = p - p1
            u_proj = quantize(np.dot(rel, u) / d, q)
            v_proj = quantize(np.dot(rel, v) / d, q)
            table[(u_proj, v_proj)].append((i, j, k))

    return table

def match_hash(stars1, stars2, q=0.05, min_votes=3):
    """
    Match patterns between two star sets using geometric hashing.

    Args:
        stars1, stars2 (list): Star data from image 1 and 2.
        q (float): Quantization step.
        min_votes (int): Vote threshold to consider a match.

    Returns:
        list of dict: Valid pattern matches.
    """
    htable = geometric_hash(stars1, q)
    coords2 = np.array([(s["x"], s["y"]) for s in stars2])
    votes = defaultdict(int)
    matches = defaultdict(list)

    for i, j in itertools.combinations(range(len(coords2)), 2):
        p1, p2 = coords2[i], coords2[j]
        vec = p2 - p1
        d = np.linalg.norm(vec)
        if d < 10:
            continue
        u = vec / d
        v = np.array([-u[1], u[0]])

        for k, p in enumerate(coords2):
            if k in (i, j): continue
            rel = p - p1
            u_proj = quantize(np.dot(rel, u) / d, q)
            v_proj = quantize(np.dot(rel, v) / d, q)
            key = (u_proj, v_proj)

            if key in htable:
                for i1, j1, k1 in htable[key]:
                    key_vote = (i1, j1, i, j)
                    votes[key_vote] += 1
                    matches[key_vote].append((k1, k))

    results = []
    for key, count in votes.items():
        if count >= min_votes:
            i1, j1, i2, j2 = key
            results.append({
                "basis_stars1": (i1, j1),
                "basis_stars2": (i2, j2),
                "vote_count": count,
                "feature_matches": matches[key]
            })

    return sorted(results, key=lambda x: -x["vote_count"])

def draw_matches(img1_path, img2_path, stars1, stars2, matches,
                 filename="matches.jpg", coords_filename="matched_coordinates.txt"):
    """
    Draw matches between two images and save results.

    Args:
        img1_path, img2_path: Paths to the two images.
        stars1, stars2: Detected stars in each image.
        matches: Output from match_hash.
        filename: Output image file.
        coords_filename: Output text file with coordinate matches.

    Returns:
        set of tuple: Unique matched (x1, y1, x2, y2) pairs.
    """
    img1 = read_color(img1_path)
    img2 = read_color(img2_path)

    # Scale both to same height
    target_height = max(img1.shape[0], img2.shape[0])
    img1_resized = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), target_height))
    img2_resized = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), target_height))

    w_offset = img1_resized.shape[1]
    combo = np.zeros((target_height, w_offset + img2_resized.shape[1], 3), dtype=np.uint8)
    combo[:, :w_offset] = img1_resized
    combo[:, w_offset:] = img2_resized

    matched_set = set()

    for match in matches[:5]:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for i1, i2 in match["feature_matches"]:
            scale1 = target_height / img1.shape[0]
            scale2 = target_height / img2.shape[0]
            x1 = int(stars1[i1]["x"] * scale1)
            y1 = int(stars1[i1]["y"] * scale1)
            x2 = int(stars2[i2]["x"] * scale2)
            y2 = int(stars2[i2]["y"] * scale2)
            cv2.line(combo, (x1, y1), (x2 + w_offset, y2), color, 2)  # Bold line
            matched_set.add((x1, y1, x2, y2))

    # Save match visualization
    cv2.imwrite(os.path.join(RESULTS_DIR, filename), combo)

    # Save only coordinates (no header)
    with open(os.path.join(RESULTS_DIR, coords_filename), "w") as f:
        for x1, y1, x2, y2 in matched_set:
            f.write(f"{x1}, {y1} -> {x2}, {y2}\n")

    return matched_set

def run_pipeline(img1_path, img2_path):
    """
    Run full detection + matching pipeline on two images.

    Args:
        img1_path (str): First image path.
        img2_path (str): Second image path.
    """
    ensure_results_dir()
    stars1 = detect_stars(img1_path)
    stars2 = detect_stars(img2_path)

    save_star_image(img1_path, stars1, "stars1_marked.jpg")
    save_star_image(img2_path, stars2, "stars2_marked.jpg")

    matches = match_hash(stars1, stars2)
    matched_set = draw_matches(img1_path, img2_path, stars1, stars2, matches,
                               filename="matches.jpg", coords_filename="matched_coordinates.txt")

    # Final summary printed to console
    print(f"[INFO] Stars detected: Image 1 -> {len(stars1)}, Image 2 -> {len(stars2)}")
    print(f"[INFO] Unique star matches written: {len(matched_set)}")

# Add the entry point
if __name__ == "__main__":
    run_pipeline("photos/fr1.jpg", "photos/ST_db2.png")
