# Geometric Star Matcher

This project detects and matches star-like features in astronomical images using geometric hashing. It identifies bright points in two input images and finds matching spatial patterns by comparing triangle configurations.

## How It Works

### Detection

Bright star-like blobs are detected in each image using OpenCV’s `SimpleBlobDetector`.

### Geometric Hashing

Triangle patterns are generated from the detected points and projected into a local coordinate system. Quantized projections are used as hash keys. Matching is done by voting on repeated triangle patterns.

### Visualization

Matching points are connected visually with lines and labeled. All output is saved in a dedicated `results_<image1>_<image2>` folder.

---

## Project Structure

.

├── star_matcher/ # Python package containing all source code

│ ├── init.py # (Optional) Marks this directory as a package

│ ├── main.py # Entry point: runs the full pipeline

│ ├── detect_stars.py # Star detection using blob detection

│ ├── match_stars.py # Geometric hashing pattern matcher

│ └── io_utils.py # Drawing and file I/O functions

├── photos/ # Folder for input images

├── results/ # Auto-generated output directories

│ └── results_fr1_ST_db2/ # Example output folder with matches

│ └── matches.jpg # Visual result showing matched stars

├── README.md 

---

## Usage

### Match a Pair of Images

Run the main script with two image paths:

```bash
python -m star_matcher.main photos/fr1.jpg photos/ST_db1.png
```

---

## Requirements

To run this project, you need the following Python packages:

- `opencv-python`
- `numpy`

Install them using pip:

```bash
pip install opencv-python numpy

```

## Example Output

After running the matcher, the following output is generated:

### Input Images:

These should be placed inside the `photos/` folder:

- `fr1.jpg`
- `ST_db1.png`

### Result:

The matched stars are connected with colored lines:

![Sample Match Output](results/results_fr1_ST_db2/matches.jpg)

> This image shows a successful match between detected star patterns in both images.

### Files Generated:

- `results_fr1_ST_db1/matched.png` – Visual match with lines
- `results_fr1_ST_db1/image1_marked.png` – Input 1 with detected stars
- `results_fr1_ST_db1/image2_marked.png` – Input 2 with detected stars
- `results_fr1_ST_db1/matched_coordinates.txt` – Coordinates of matches
