import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_points(filename):
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
    return np.array(points)

def write_points(points, filename):
    im_name = os.path.splitext(os.path.basename(filename))[0]
    im_name = im_name.split('.')[0]
    points_filename = os.path.join("points", f"{im_name}.points")

    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    with open(points_filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")

    print(f"Points saved to {points_filename}")

def pick_points(im_path):
    print('Please click on the image to select points. Press Enter when finished.')

    if not os.path.exists(im_path):
        print(f"Error: The file '{im_path}' does not exist.")
        sys.exit(1)

    im = plt.imread(im_path)
    height, width = im.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(im)

    points = []
    
    while True:
        point = plt.ginput(1, timeout=-1)
        if not point:
            break
        x, y = point[0]
        normalized_point = (x / width, y / height)
        points.append(normalized_point)
        ax.plot(x, y, 'ro', markersize=5)
        fig.canvas.draw()

    plt.close()

    write_points(points, im_path)

    return points

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <image_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    pick_points(image_path)

if __name__ == "__main__":
    main()