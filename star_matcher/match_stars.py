import numpy as np
from collections import defaultdict
from itertools import combinations

class MatchRecord:
    """
    Represents a pattern match between two images.
    Stores the basis pair indices from both sets, number of votes, and point matches.
    """
    def __init__(self, ref_basis, target_basis, votes, matched_points):
        self.ref_basis = ref_basis            # Tuple of indices (i, j) in the reference image
        self.target_basis = target_basis      # Tuple of indices (i, j) in the target image
        self.votes = votes                    # Number of matching triangles that support this match
        self.matched_points = matched_points  # List of matched point index pairs (ref_k, target_k)


class GeoHasher:
    """
    Implements geometric hashing to find matching patterns between two point sets.
    """
    def __init__(self, res=0.05, min_dist=10):
        self.res = res              # Quantization resolution for the projected coordinates
        self.min_dist = min_dist    # Minimum distance between basis points to form a valid basis

    def quantize(self, value):
        """
        Snap a continuous value to the nearest discrete grid line based on resolution.
        """
        return round(value / self.res) * self.res

    def _basis(self, p1, p2):
        """
        Construct an orthonormal basis from two points if their distance is above threshold.

        Returns:
            x_axis: normalized vector from p1 to p2
            y_axis: perpendicular vector to x_axis (90 degrees CCW)
            dist: Euclidean distance between p1 and p2
        """
        vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        dist = np.linalg.norm(vec)
        if dist < self.min_dist:
            return None, None, None
        x_axis = vec / dist
        y_axis = np.array([-x_axis[1], x_axis[0]]) 
        return x_axis, y_axis, dist

    def _project(self, pt, origin, x_axis, y_axis, dist):
        """
        Project a point into a local coordinate system defined by (origin, x_axis, y_axis),
        then quantize the result to form a hash key.
        """
        rel = pt - origin
        return (
            self.quantize(np.dot(rel, x_axis) / dist),  # Project onto x-axis and normalize
            self.quantize(np.dot(rel, y_axis) / dist)   # Project onto y-axis and normalize
        )

    def match(self, ref_points, target_points, min_votes=3):
        """
        Perform geometric matching between reference and target points.

        Builds a hash table from the reference image based on triangle patterns.
        Then searches the target image for similar patterns and accumulates votes.

        Returns:
            A list of MatchRecord objects, sorted by vote count (descending).
        """
        db = self._build_db(ref_points)  # Build geometric hash DB from reference image

        coords = np.array([(p.x, p.y) for p in target_points])  # Target image points
        votes = defaultdict(int)         # Key: (ref_i, ref_j, target_i, target_j)
        matches = defaultdict(list)      # Store supporting point matches for each key


        # Iterates over every unique pair of points (i, j) from the list of coordinates.
        for i, j in combinations(range(len(coords)), 2):
            x_axis, y_axis, dist = self._basis(coords[i], coords[j])   
            if x_axis is None:
                continue
            
            #Iterates over every third point k that is not i or j.
            for k, pt in enumerate(coords):
                if k in (i, j):
                    continue

                # Project third point k into local coordinate system of (i, j)
                key = self._project(pt, coords[i], x_axis, y_axis, dist)

                # See if this pattern exists in the reference DB
                for ref_i, ref_j, ref_k in db.get(key, []):
                    vote_key = (ref_i, ref_j, i, j)
                    votes[vote_key] += 1
                    matches[vote_key].append((ref_k, k))

        # Build final result list from vote count
        results = []
        for key, count in votes.items():
            if count >= min_votes:
                results.append(MatchRecord(
                    ref_basis=key[:2],
                    target_basis=key[2:],
                    votes=count,
                    matched_points=matches[key]
                ))

        return sorted(results, key=lambda m: -m.votes)  # Sort matches by strength

    def _build_db(self, points):
        """
        Build a hash table of relative triangle patterns from the reference points.
        Keys are quantized projections; values are triples of point indices (i, j, k).
        """
        coords = np.array([(p.x, p.y) for p in points])
        db = defaultdict(list)

        # For every pair of points (i, j), define a coordinate system
        for i, j in combinations(range(len(coords)), 2):
            x_axis, y_axis, dist = self._basis(coords[i], coords[j])
            if x_axis is None:
                continue

            # Project every third point (k) into this local basis
            for k, pt in enumerate(coords):
                if k in (i, j):
                    continue
                key = self._project(pt, coords[i], x_axis, y_axis, dist)
                db[key].append((i, j, k))

        return db


def match_star_patterns(points1, points2):
    """
    Perform geometric star pattern matching using GeoHasher.
    """
    return GeoHasher().match(points1, points2)
