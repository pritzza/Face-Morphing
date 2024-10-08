import cv2
import numpy as np
import os

from PIL import Image
from scipy.spatial import Delaunay

from point_reader import write_points, read_points

def extrapolate(a, b, factor):
    return a + factor * (b - a)

def lerp(a, b, t):
    return (1.0 - t) * a + t * b

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

def warp_img(img, start_points, end_points, end_triangulation):
    img_out = img.copy()
    
    # warp one triangle at a time
    for simplex in end_triangulation:
        
        start_tri = start_points[simplex]
        end_tri = end_points[simplex]

        s_bb = cv2.boundingRect(np.float32([start_tri]))
        e_bb = cv2.boundingRect(np.float32([end_tri]))

        # adjust triangle points to bounding box coords
        start_tri_cropped = [(start_tri[i][0] - s_bb[0], start_tri[i][1] - s_bb[1]) for i in range(3)]
        end_tri_cropped   = [  (end_tri[i][0] - e_bb[0],   end_tri[i][1] - e_bb[1]) for i in range(3)]
        mask = np.zeros((e_bb[3], e_bb[2], 3))
        cv2.fillConvexPoly(mask, np.int32(end_tri_cropped), (1, 1, 1), 16, 0)

        # crop img to bounding box
        img_cropped = img[s_bb[1]:s_bb[1] + s_bb[3], s_bb[0]:s_bb[0] + s_bb[2]]
        img_warped = affine_transform(img_cropped, start_tri_cropped, end_tri_cropped, (e_bb[2], e_bb[3]))

        # blend
        img_out[e_bb[1]:e_bb[1]+e_bb[3], e_bb[0]:e_bb[0]+e_bb[2]] = \
            img_out[e_bb[1]:e_bb[1]+e_bb[3], e_bb[0]:e_bb[0]+e_bb[2]] * (1 - mask) + img_warped * mask

    return img_out

def get_affine_transform(start_tri, end_tri):

    x1, y1 = start_tri[0]
    x2, y2 = start_tri[1]
    x3, y3 = start_tri[2]
    x1_prime, y1_prime = end_tri[0]
    x2_prime, y2_prime = end_tri[1]
    x3_prime, y3_prime = end_tri[2]

    A = np.array([
        [x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1],
        [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
        [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]
    ])
    B = np.array([x1_prime, y1_prime, x2_prime, y2_prime, x3_prime, y3_prime])

    # solve for affine transform parameters
    affine_params = np.linalg.solve(A, B)
    return affine_params.reshape(2, 3)

def affine_transform(img, start_tri, end_tri, size):
    M = get_affine_transform(start_tri, end_tri)
    return cv2.warpAffine(img, M, size, borderMode=cv2.BORDER_REFLECT_101)

def morph_movie(total_frames, duration, images, points, dim, out_name):
    assert len(images) == len(points), "Each image must have corresponding points."

    if os.path.dirname(out_name) == '':
        out_name = os.path.join('.', out_name)
    
    os.makedirs(os.path.dirname(out_name), exist_ok=True)

    frames = []

    # standardize image size and points
    resized_images = [cv2.resize(img, dim) for img in images]
    normalized_points = [normalize_points(lm, dim) for lm in points]

    # go over each pair of images
    for i in range(len(resized_images) - 1):
        img1, img2 = np.float32(resized_images[i]), np.float32(resized_images[i + 1])
        img1_points = scale_points(normalized_points[i], dim)
        img2_points = scale_points(normalized_points[i + 1], dim)

        avg_points = (img1_points + img2_points) / 2
        triangulation = Delaunay(avg_points).simplices

        # morph
        for j in range(total_frames):
            t = j / (total_frames - 1)
            weighted_points = lerp(img1_points, img2_points, t)
            img1_warped = warp_img(img1, img1_points, weighted_points, triangulation)
            img2_warped = warp_img(img2, img2_points, weighted_points, triangulation)
            blended = lerp(img1_warped, img2_warped, t)

            frame = Image.fromarray(cv2.cvtColor(np.uint8(blended), cv2.COLOR_BGR2RGB))
            frames.append(frame)

    gif_filename = f'{out_name}.gif'

    frames[0].save(gif_filename, 
                   append_images=frames[1:],
                   duration=duration,
                   save_all=True, 
                   loop=0
                   )

def load_dataset(dataset_path):
    images = []
    points = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dataset_path, filename)
            pts_path = os.path.join(dataset_path, filename[:-4] + '.pts')
            asf_path = os.path.join(dataset_path, filename[:-4] + '.asf')
            
            p = None

            if os.path.exists(pts_path):
                p = read_points(pts_path)  # Read from .pts file
            elif os.path.exists(asf_path):
                p = read_points(asf_path)  # Read from .asf file

            img = cv2.imread(img_path)

            if img is not None and p is not None:
                images.append(img)
                points.append(p)
            else:
                print(f"Error: Could not read image '{img_path}'")
    
    return images, points

def compute_average_shape(points):
    return np.mean(points, axis=0)

def warp_to_average_shape(images, points, avg_shape, triangulation):
    warped_images = []
    for img, p in zip(images, points):
        try:
            warped = warp_img(img, p, avg_shape, triangulation)
            warped_images.append(warped)
        except: # catch any errors with images in the dataset
            pass
    return warped_images

def compute_average_face(dataset_path, dim, out_name):
    images, points = load_dataset(dataset_path)
    avg_shape = compute_average_shape(points)
    write_points(normalize_points(avg_shape, dim), out_name + '.points')
    triangulation = Delaunay(avg_shape).simplices
    warped_images = warp_to_average_shape(images, points, avg_shape, triangulation)
    avg_face = np.mean(warped_images, axis=0).astype(np.uint8)
    cv2.imwrite(out_name + '.jpg', avg_face)

def warp_to_average_face(img_path, avg_face_path, points_path, avg_shape_path, out_name):

    img = cv2.imread(img_path)
    dim = (img.shape[1], img.shape[0])
    img_points = read_points(points_path, dim)
        
    avg_face = cv2.imread(avg_face_path)
    avg_shape= read_points(avg_shape_path, dim)
    triangulation = Delaunay(avg_shape).simplices

    img_to_avg = warp_img(img, img_points, avg_shape, triangulation)
    avg_to_img = warp_img(avg_face, avg_shape, img_points, triangulation)

    cv2.imwrite(out_name + '_forward.jpg', img_to_avg)
    cv2.imwrite(out_name + '_backward.jpg', avg_to_img)

def main():

    COMPUTE_AVG = True
    WARP_TO_AVG = True
    MORPH = True
    CARICATURE = True

    BR_PATH = "res/faces/"
    BR_DIM = (250,300)

    if COMPUTE_AVG:
        compute_average_face(BR_PATH, BR_DIM, 'avg_face')

    if WARP_TO_AVG:
        warp_to_average_face('res/me.png', 
                                'res/avg_happy_man.jpg', 
                                'points/me.points', 
                                'points/avg_happy_man.points', 
                                'brazilian_me')

    if MORPH:

        img_names = [
            'noa.jpg',
            'wolf.jpg',
            'noa.jpg',
        ]
        points_names = [
            'noa.points',
            'wolf.points',
            'noa.points',
        ]

        dim = (500,600)
        out_name = 'morph'

        imgs = [cv2.imread("res/" + i) for i in img_names]
        points = [read_points("points/" + p) for p in points_names]

        frames = 30
        duration = 30
        morph_movie(frames, duration, imgs, points, dim, out_name)

    if CARICATURE:
        img_path = 'res/me.png'
        points_path = 'points/me.points'
        caricature_shape_path = 'points/avg_happy_man.points'

        out_name = 'caricature'

        img = cv2.imread(img_path)
        dim = (img.shape[1], img.shape[0])
        img_points = read_points(points_path, dim)
        caricature_shape = read_points(caricature_shape_path, dim)
        
        extrapolated_points = extrapolate(img_points, caricature_shape, factor=1.5)
        triangulation = Delaunay(extrapolated_points).simplices
        caricature_img = warp_img(img, img_points, extrapolated_points, triangulation)
        
        cv2.imwrite(out_name + '.jpg', caricature_img)

if __name__ == "__main__":
    main()