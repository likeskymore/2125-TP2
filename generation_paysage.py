import numpy as np
from solid import *
from solid.utils import *
import random

# Generate elevation for one island
def generate_elevation(size=80, island_radius=30, max_peak_height=15, center_x=None, center_y=None):
    elevation = np.zeros((size, size))  # Create a grid filled with zeros (flat)

    # Choose a random center if not provided
    if center_x is None:
        center_x = random.randint(0 + island_radius // 2, island_radius)
    if center_y is None:
        center_y = random.randint(0 + island_radius // 2, island_radius)

    # Fill elevation values based on distance from the center (higher in the middle)
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance < island_radius:
                random_factor = random.uniform(0.7, 1.3)
                peak_height = np.clip(
                    (max_peak_height - (distance / island_radius) * max_peak_height) * random_factor,
                    0, max_peak_height
                )
                elevation[i, j] = peak_height
            else:
                elevation[i, j] = 0

    # Smooth the terrain a bit
    for _ in range(2):
        smoothed = np.copy(elevation)
        for i in range(1, size-1):
            for j in range(1, size-1):
                smoothed[i, j] = np.mean(elevation[i-1:i+2, j-1:j+2])
        elevation = smoothed

    return elevation

# Apply smoothing to elevation (blurs the height map)
def smooth_elevation(elevation, iterations=3, kernel_size=3):
    size = elevation.shape[0]
    smoothed = np.copy(elevation)

    # Use a small average kernel to smooth
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    for _ in range(iterations):
        new_smoothed = np.copy(smoothed)
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                region = smoothed[i - 1:i + 2, j - 1:j + 2]
                new_smoothed[i, j] = np.sum(region * kernel)
        smoothed = new_smoothed

    return smoothed

# Add a river from one point to another
def add_river(elevation, start=(10, 10), end=(70, 70), width=1, wiggle=20):
    path = []
    current = np.array(start, dtype=float)
    direction = np.array(end) - current
    direction = direction / np.linalg.norm(direction)

    for _ in range(100):
        deviation = np.random.uniform(-wiggle, wiggle, size=2)
        current += direction + deviation * 0.1
        x, y = int(round(current[0])), int(round(current[1]))
        if 0 <= x < elevation.shape[0] and 0 <= y < elevation.shape[1]:
            path.append((x, y))

    # Lower the elevation along the river path
    for x, y in path:
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < elevation.shape[0] and 0 <= ny < elevation.shape[1]:
                    elevation[nx, ny] = -1

    return elevation, path

# Add lakes at random places surround by land, avoiding river
def generate_lakes(elevation, lake_count=3, lake_size_min=5, lake_size_max=8, min_distance=10, river_path=None):
    size = elevation.shape[0]
    lake_centers = []
    attempts = 0

    # Check if a circle is surrounded by land
    def is_surrounded_by_land(x, y, radius):
        for i in range(x - radius - 1, x + radius + 2):
            for j in range(y - radius - 1, y + radius + 2):
                if 0 <= i < size and 0 <= j < size:
                    distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    if radius < distance <= radius + 1:
                        if elevation[i, j] < 0:
                            return False
                else:
                    return False
        return True

    # Try placing lakes
    while len(lake_centers) < lake_count and attempts < 1000:
        attempts += 1
        lake_center_x = random.randint(lake_size_max, size - lake_size_max)
        lake_center_y = random.randint(lake_size_max, size - lake_size_max)
        lake_radius = random.randint(lake_size_min, lake_size_max)

        if river_path:
            too_close = any(np.hypot(lake_center_x - rx, lake_center_y - ry) < lake_radius + 3 for rx, ry in river_path)
            if too_close:
                continue

        if all(np.hypot(lake_center_x - cx, lake_center_y - cy) > (min_distance + lake_radius) for cx, cy in lake_centers):
            if not is_surrounded_by_land(lake_center_x, lake_center_y, lake_radius):
                continue

            # Lower the elevation in lake area
            for i in range(lake_center_x - lake_radius, lake_center_x + lake_radius + 1):
                for j in range(lake_center_y - lake_radius, lake_center_y + lake_radius + 1):
                    if 0 <= i < size and 0 <= j < size:
                        distance = np.sqrt((i - lake_center_x) ** 2 + (j - lake_center_y) ** 2)
                        if distance <= lake_radius:
                            elevation[i, j] = -2

            lake_centers.append((lake_center_x, lake_center_y))

    return elevation

# Check if a point is near water
def is_near_water(i, j, elevation, threshold=1.5):
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = i + di, j + dj
            if 0 <= ni < elevation.shape[0] and 0 <= nj < elevation.shape[1]:
                if elevation[ni, nj] < threshold and elevation[ni, nj] != -1:
                    return True
    return False

# Detect if a point is a cliff
def is_cliff(i, j, elevation, cliff_threshold=5):
    height = elevation[i, j]
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < elevation.shape[0] and 0 <= nj < elevation.shape[1]:
                neighbors.append(elevation[ni, nj])
    return any(abs(height - n) > cliff_threshold for n in neighbors)

# Add trees in suitable elevation zones
def place_vegetation(elevation, threshold=5, density=0.1, cluster_size_min=5, cluster_size_max=15, max_attempts=1000):
    trees = []
    size = elevation.shape[0]

    # Cluster treees to make forest
    def generate_forest(x, y, cluster_size):
        cluster_trees = []
        for _ in range(cluster_size):
            # Randomly offset each tree position within the cluster area
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            nx, ny = x + offset_x, y + offset_y
            if 0 <= nx < size and 0 <= ny < size:
                h = elevation[nx, ny]
                if threshold < h < 12 and random.random() < density:
                    # Create the trunk (1.5-2 blocks tall)
                    trunk_height = random.uniform(1.5,2)  
                    trunk_radius = random.uniform(0.05, 0.1)  
                    # Trunk starts at the terrain height and goes upwards
                    trunk = translate([nx + 0.5, ny + 0.5, h + trunk_height / 2])(
                        color([0.4, 0.2, 0.1])(
                            cylinder(h=trunk_height, r1=trunk_radius, r2=trunk_radius)  
                        )
                    )

                    # Add leaves for the tree trunk
                    canopy_height = random.uniform(1, 2) 
                    canopy_base_radius = random.uniform(0.3, 0.6)  
                    canopy_top_radius = random.uniform(0.1, 0.3)  
                    # Place leaves just above the trunk
                    canopy = translate([nx + 0.5, ny + 0.5, h + trunk_height + canopy_height / 2])(
                        color([0.1, 0.4, 0.1])(
                            cylinder(h=canopy_height, r1=canopy_base_radius, r2=canopy_top_radius)  
                        )
                    )

                    # Add both the trunk and the leaves to the tree
                    cluster_trees.append(trunk)
                    cluster_trees.append(canopy)

        return cluster_trees

    # Try placing tree clusters
    attempts = 0
    while attempts < max_attempts:
        # Choose a random starting point for a tree cluster
        x = random.randint(1, size - 2)
        y = random.randint(1, size - 2)
        h = elevation[x, y]
        
        if threshold < h < 12 and random.random() < density:
            # Randomly choose the size of the cluster
            cluster_size = random.randint(cluster_size_min, cluster_size_max)
            cluster_trees = generate_forest(x, y, cluster_size)
            trees.extend(cluster_trees)
            attempts += 1

    return trees

# Create a combined terrain from many small islands
def generate_combined_elevation(size=80, island_radius=12, num_islands=10, min_distance=10):
    elevation = np.zeros((size, size))
    centers = []

    attempts = 0
    while len(centers) < num_islands and attempts < 1000:
        attempts += 1
        center_x = random.randint(island_radius, size - island_radius)
        center_y = random.randint(island_radius, size - island_radius)
        distance_factor = min_distance + random.randint(-5, 10)
        if all(np.hypot(center_x - cx, center_y - cy) >= distance_factor for cx, cy in centers):
            max_peak_height = random.randint(4, 14)
            radius_x = island_radius + random.randint(-3, 5)
            radius_y = island_radius + random.randint(-3, 5)

            for i in range(size):
                for j in range(size):
                    dx = (i - center_x) / radius_x
                    dy = (j - center_y) / radius_y
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < 1:
                        rf = random.uniform(0.7, 1.3)
                        h = np.clip((max_peak_height - distance * max_peak_height) * rf, 0, max_peak_height)
                        elevation[i, j] = max(elevation[i, j], h)

            centers.append((center_x, center_y))
    
    elevation = smooth_elevation(elevation)                        # Apply Smoothing
    elevation, river_path = add_river(elevation)                   # Add river
    elevation = generate_lakes(elevation, river_path=river_path)   # Add lakes
    return elevation


# Final 3D scene creation
def generate_island():
    elevation = generate_combined_elevation()
    island = []

    # Add ocean base with engraved text
    island.append(translate([0, 0, 0])(
        difference()(
            color([0, 0.3, 1])(
                cube([80, 80, 1], center=False)
            ),
            translate([60, 3, -0.5])(
                linear_extrude(height=1)( 
                    mirror([1, 0, 0])(
                        text("T-et-M-IFT2125", size=6, font="Arial", halign="true")
                    )
                )
            )
        )
    ))

    # Create colored terrain blocks
    for i in range(80):
        for j in range(80):
            height = elevation[i, j]
            if height <= 0.5 and is_near_water(i, j, elevation) and not is_cliff(i, j, elevation):
                color_rgb = [0.9, 0.8, 0.5]  # Beach
            elif is_cliff(i, j, elevation):
                color_rgb = [0.5, 0.4, 0.4]  # Cliff
            else:
                color_rgb = [0, 1, 0]

            island.append(
                color(color_rgb)(
                    translate([i, j, 1])(
                        cube([1, 1, height], center=False)
                    )
                )
            )

    # Add vegetation (trees)
    trees = place_vegetation(elevation)
    island.extend(trees)

    return island

# Render and save to .scad file
island = generate_island()
final_island = union()(*island)
scad_render_to_file(final_island, 'paysage.scad')
