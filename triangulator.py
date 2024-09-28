import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from point_picker import read_points

def triangulate_points(points):
    # todo
    return Delaunay(points)

def plot_triangulation(points, tri):
    plt.figure(figsize=(10, 10))
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.title('Delaunay Triangulation of Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python triangulate_points.py <points_filename>")
        sys.exit(1)

    points_filename = sys.argv[1]
    
    points = read_points(points_filename)    
    triangles = triangulate_points(points)
    plot_triangulation(points, triangles)
    
    print(str(len(triangles.simplices)) + " triangles made")

if __name__ == "__main__":
    main()