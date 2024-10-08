import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import numpy as np

def read_points(filename, dim=(1,1)):

    if filename.endswith('.points'):
        return parse_points(filename, dim)
    elif filename.endswith('.pts'):
        return parse_pts(filename)

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

    # Convert points to numpy array for efficient operations
    points = np.array(points)

    # Scale points to image dimensions
    points[:, 0] *= dim[0]
    points[:, 1] *= dim[1]
    
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
    
    print(coordinates)

    return np.array(coordinates)

# Example usage:
# file_path = 'path/to/your/file.txt'
# coordinates = read_coordinates_file(file_path)
# print(coordinates)

def write_points(points, filename):
    im_name = os.path.splitext(os.path.basename(filename))[0]
    im_name = im_name.split('.')[0]
    points_filename = os.path.join("points", f"{im_name}.points")

    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    with open(points_filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")

    print(f"Points saved to {points_filename}")

class PointEditor:
    def __init__(self, image_path, points):
        self.image_path = image_path
        self.points = points
        self.selected_point = None
        self.fig, self.ax = plt.subplots()
        self.image = plt.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.imshow(self.image)
        x = self.points[:, 0] * self.width
        y = self.points[:, 1] * self.height
        self.scatter = self.ax.scatter(x, y, c='r', s=50)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        cont, ind = self.scatter.contains(event)
        if cont:
            self.selected_point = ind['ind'][0]

    def on_motion(self, event):
        if self.selected_point is None:
            return
        if event.inaxes != self.ax:
            return
        self.points[self.selected_point] = (event.xdata / self.width, event.ydata / self.height)
        self.update_plot()

    def on_release(self, event):
        self.selected_point = None

    def on_key(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    def update_plot(self):
        x = self.points[:, 0] * self.width
        y = self.points[:, 1] * self.height
        self.scatter.set_offsets(np.c_[x, y])
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()

def edit_points(image_path, points_path):
    
    if os.path.exists(points_path):
        points = read_points(points_path)
    else:
        print(f"No existing points file found. Starting with an empty set of points.")
        points = np.array([])

    editor = PointEditor(image_path, points)
    editor.run()

    write_points(editor.points, points_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python .\point_editor.py <image_filename> <points_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    points_path = sys.argv[2]
    edit_points(image_path, points_path)

if __name__ == "__main__":
    main()