import numpy as np
import os

def is_normalized(points):
    points_array = np.array(points)
    return np.all((points_array >= 0) & (points_array <= 1))

def normalize_points(points, dim):

    points = np.array(points)

    if is_normalized(points):
        return points
    
    points[:, 0] *= 1/dim[0]
    points[:, 1] *= 1/dim[1]

    return points

def scale_points(points, dim):
    points = np.array(points)
    points[:, 0] *= dim[0]
    points[:, 1] *= dim[1]
    return points

def read_points(filename, dim=(1,1)):

    if filename.endswith('.points'):
        return parse_points(filename, dim)
    elif filename.endswith('.pts'):
        return parse_pts(filename)
    elif filename.endswith('.asf'):
        return parse_asf(filename, dim)

    print("error: unknown file format")

def parse_points(filename, dim=(1,1)):

    points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split(','))
                points.append((x, y))
        print(f"Successfully read {len(points)} points from {filename}")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    points = np.array(points)

    # small
    sml = 1.0 - 1e-4

    # moints for corners and edge midpoints
    extra_points = np.array([
        [0.0, 0.0], [sml, 0.0], [0.0, sml], [sml, sml],  # corners
        [0.5, 0.0], [0.0, 0.5], [sml, 0.5], [0.5, sml]   # edge midpoints
    ])

    # dont add duplicate points
    for point in extra_points:
        if not any(np.allclose(point, p) for p in points):
            points = np.vstack([points, point])

    points = scale_points(points, dim)

    return points

def parse_pts(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    coordinates = []
    
    for line in lines:
        if line.startswith('version') or line.startswith('n_points') or line.startswith('{') or line.startswith('}'): 
            continue
        point = tuple(map(float, line.split()))
        coordinates.append(point)
    
    return np.array(coordinates)

def parse_asf(filename, dim=(1, 1)):
    points = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Start parsing from the 17th line onward
        for line in lines[17:]:
            # Split line by spaces and filter empty strings
            data = line.strip().split()
            if len(data) < 4:
                continue  # Skip lines that do not contain enough data

            # Extract the third and fourth values (x_rel, y_rel)
            try:
                x_rel = float(data[2])
                y_rel = float(data[3])
                points.append((x_rel, y_rel))
            except ValueError:
                # Handle lines that may not contain valid floats
                continue

        # Convert points to a numpy array
        points = np.array(points)

        # Optionally scale the points to the given dimensions (dim)
        points = scale_points(points, dim)
        print(f"Successfully read {len(points)} points from {filename}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return points

def write_points(points, filename):
    im_name = os.path.splitext(os.path.basename(filename))[0]
    im_name = im_name.split('.')[0]
    points_filename = os.path.join("points", f"{im_name}.points")

    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    with open(points_filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")

    print(f"Points saved to {points_filename}")

