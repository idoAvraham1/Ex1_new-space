from detect_stars import detect_stars
from match_stars import match_star_patterns
from io_utils import load_image, save_image, save_text, draw_detected_points, draw_matches
import os

RESULTS_DIR = "results"

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    """
    Invokes the star matching pipeline: detection, matching, and visualization.
    """
    ensure_results_dir()

    # Input image paths
    img1_path = "photos/ST_db2.png"
    img2_path = "photos/fr2.jpg"

    # Load original images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Detect stars
    points1 = detect_stars(img1_path)
    points2 = detect_stars(img2_path)

    print(f"[INFO] Detected: {len(points1)} points in img1, {len(points2)} in img2")

    # Save marked images
    save_image(draw_detected_points(img1, points1), f"{RESULTS_DIR}/marked_1.jpg")
    save_image(draw_detected_points(img2, points2), f"{RESULTS_DIR}/marked_2.jpg")

    # Match and visualize
    matches = match_star_patterns(points1, points2)
    if matches:
        match_img, lines = draw_matches(img1, img2, points1, points2, matches)
        save_image(match_img, f"{RESULTS_DIR}/matches.jpg")
        save_text(lines, f"{RESULTS_DIR}/matches.txt")
        print(f"[INFO] Matches found: {len(lines.splitlines())}")
    else:
        print("[INFO] No matches found.")

if __name__ == "__main__":
    main()