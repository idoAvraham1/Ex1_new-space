from star_matcher.detect_stars import detect_stars
from star_matcher.match_stars import match_star_patterns
from star_matcher.io_utils import load_image, save_image, save_text, draw_detected_points, draw_matches
import os
import sys
import os.path as osp

def ensure_dir(path):
    """
    Ensure the output directory exists.
    """
    os.makedirs(path, exist_ok=True)

def clean_name(path):
    """
    Extract base file name without extension.
    Used to name the output directory.
    """
    return osp.splitext(osp.basename(path))[0]

def main(img1_path, img2_path):
    """
    Invokes the star matching pipeline: detection, matching, and visualization.
    Takes two image paths as input and creates a results_<image1>_<image2> directory.
    """
    out_dir = f"results_{clean_name(img1_path)}_{clean_name(img2_path)}"
    ensure_dir(out_dir)

    # Load original images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Detect stars
    points1 = detect_stars(img1_path)
    points2 = detect_stars(img2_path)

    print(f"[INFO] Detected: {len(points1)} points in img1, {len(points2)} in img2")

    # Save marked images
    save_image(draw_detected_points(img1, points1), f"{out_dir}/marked_1.jpg")
    save_image(draw_detected_points(img2, points2), f"{out_dir}/marked_2.jpg")

    # Match and visualize
    matches = match_star_patterns(points1, points2)
    if matches:
        match_img, lines = draw_matches(img1, img2, points1, points2, matches)
        save_image(match_img, f"{out_dir}/matches.jpg")
        save_text(lines, f"{out_dir}/matches.txt")
        print(f"[INFO] Matches found: {len(lines.splitlines())}")
    else:
        print("[INFO] No matches found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <image1_path> <image2_path>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    main(img1_path, img2_path)
