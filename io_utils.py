import cv2
import numpy as np
import os

def load_image(path):
   
    return cv2.imread(path)

def save_image(image, path):
    
    cv2.imwrite(path, image)

def save_text(text, path):
   
    with open(path, 'w') as f:
        f.write(text)

def draw_detected_points(image, points):
    """
    Draw circles and (x, y) labels for detected points on the image.
    """
    out = image.copy()
    for p in points:
        cv2.circle(out, (p.x, p.y), int(round(p.r)), (0, 255, 0), 1)  # Green circle
        cv2.putText(out, f"{p.x},{p.y}", (p.x + 4, p.y - 4),          # Small label
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return out

def draw_matches(img1, img2, pts1, pts2, matches, max_lines=5):
    """
    Create a side-by-side image of matches with lines connecting corresponding points.
    """
    # Resize both images to the same height
    h = max(img1.shape[0], img2.shape[0])
    scale1 = h / img1.shape[0]
    scale2 = h / img2.shape[0]
    img1 = cv2.resize(img1, (int(img1.shape[1] * scale1), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * scale2), h))

    # Create blank canvas to combine both images
    offset = img1.shape[1]
    combined = np.zeros((h, offset + img2.shape[1], 3), dtype=np.uint8)
    combined[:, :offset] = img1
    combined[:, offset:] = img2

    # Avoid duplicate lines
    seen1, seen2 = set(), set()
    drawn = set()

    # Draw lines for top N matches
    for match in matches[:max_lines]:
        for i, j in match.matched_points:
            if i in seen1 or j in seen2:
                continue
            seen1.add(i)
            seen2.add(j)

            p1 = pts1[i]
            p2 = pts2[j]
            x1, y1 = int(p1.x * scale1), int(p1.y * scale1)
            x2, y2 = int(p2.x * scale2), int(p2.y * scale2)

            color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color
            cv2.line(combined, (x1, y1), (x2 + offset, y2), color, 2)
            drawn.add((x1, y1, x2, y2))

    # Format matched coordinates as text
    text = "\n".join(f"{x1}, {y1} -> {x2}, {y2}" for x1, y1, x2, y2 in drawn)
    return combined, text
