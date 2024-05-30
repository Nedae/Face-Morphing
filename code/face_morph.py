import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha, img_out_1 = None, img_out_2 = None, draw_triangles=True, context = "global"):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

    if draw_triangles:
        pt1 = (int(t[0][0]), int(t[0][1]))
        pt2 = (int(t[1][0]), int(t[1][1]))
        pt3 = (int(t[2][0]), int(t[2][1]))
        cv2.line(img, pt1, pt2, (255, 255, 255), 1, 8, 0)
        cv2.line(img, pt2, pt3, (255, 255, 255), 1, 8, 0)
        cv2.line(img, pt3, pt1, (255, 255, 255), 1, 8, 0)
        
    if context == "local":
        imgRect1 = warpImage1
        imgRect2 = warpImage2
        img_out_1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_out_1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect1 * mask
        img_out_2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_out_2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect2 * mask


def generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, size, output):
    num_images = int(duration * frame_rate)
    
    # Generate palette
    palette_proc = Popen([
        'ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', f'{size[1]}x{size[0]}', '-i', '-',
        '-vf', 'palettegen', '-t', str(duration), '-y', 'palette.png'
    ], stdin=PIPE)
    
    for j in range(num_images):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        points = []
        alpha = j / (num_images - 1)
        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)
        for i in range(len(tri_list)):
            x, y, z = int(tri_list[i][0]), int(tri_list[i][1]), int(tri_list[i][2])
            t1, t2, t = [points1[x], points1[y], points1[z]], [points2[x], points2[y], points2[z]], [points[x], points[y], points[z]]
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
            pt1, pt2, pt3 = (int(t[0][0]), int(t[0][1])), (int(t[1][0]), int(t[1][1])), (int(t[2][0]), int(t[2][1]))
            cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)
        
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(palette_proc.stdin, 'JPEG')

    palette_proc.stdin.close()
    palette_proc.wait()

    # Generate GIF with palette
    gif_proc = Popen([
        'ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', f'{size[1]}x{size[0]}', '-i', '-',
        '-i', 'palette.png', '-lavfi', 'fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse',
        '-loop', '0', '-gifflags', '+transdiff', '-y', output
    ], stdin=PIPE)

    for j in range(num_images):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        points = []
        alpha = j / (num_images - 1)
        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)
        for i in range(len(tri_list)):
            x, y, z = int(tri_list[i][0]), int(tri_list[i][1]), int(tri_list[i][2])
            t1, t2, t = [points1[x], points1[y], points1[z]], [points2[x], points2[y], points2[z]], [points[x], points[y], points[z]]
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
            pt1, pt2, pt3 = (int(t[0][0]), int(t[0][1])), (int(t[1][0]), int(t[1][1])), (int(t[2][0]), int(t[2][1]))
            cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)
        
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(gif_proc.stdin, 'JPEG')

    gif_proc.stdin.close()
    gif_proc.wait()
    
    
def generate_weighted_image(img1, img2, points1, points2, tri_list, size, alpha, output_path):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    points = []

    for i in range(len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)
    for i in range(len(tri_list)):
        x, y, z = int(tri_list[i][0]), int(tri_list[i][1]), int(tri_list[i][2])
        t1, t2, t = [points1[x], points1[y], points1[z]], [points2[x], points2[y], points2[z]], [points[x], points[y], points[z]]
        morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha, draw_triangles=False)

    res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
    res.save(output_path, 'PNG')


    res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
    res.save(output_path, 'PNG')

    
 
        
def crop_and_replace(img, points, wimg1):    
    # Convert points to a numpy array
    points = np.array(points)
    
    # Filter out landmarks that are very close to the margins
    margin = 2
    filtered_points = [point for point in points if margin < point[0] < img.shape[1] - margin and margin < point[1] < img.shape[0] - margin]
    filtered_points = np.array(filtered_points)
    
    # Find the bounding box of the filtered face landmarks
    x_min = int(np.min(filtered_points[:, 0]))
    x_max = int(np.max(filtered_points[:, 0]))
    y_min = int(np.min(filtered_points[:, 1]))
    y_max = int(np.max(filtered_points[:, 1]))

    gmorph = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))

#     # Draw face landmarks on gmorph
#     img_with_landmarks = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
#     for point in points:
#         x, y = int(point[0]), int(point[1])
#         if x < 20 or y < 20 or x > 1010 or y > 1010:
#             cv2.rectangle(img_with_landmarks, (x-3, y-3), (x+3, y+3), (255, 0, 0), -1)  # Red squares for boundary points
#         else:
#             cv2.circle(img_with_landmarks, (x, y), 2, (0, 255, 0), -1)  # Green circles for regular points

#     gmorph_with_landmarks = Image.fromarray(img_with_landmarks)
#     gmorph_with_landmarks.save("results/gmorph_with_landmarks.png")
    
#     # Save the original gmorph for reference
#     gmorph.save("results/gmorph.png")


    # Crop the region from the original image
    cropped_img = img[y_min:y_max, x_min:x_max]

#     crp = Image.fromarray(cv2.cvtColor(np.uint8(cropped_img), cv2.COLOR_BGR2RGB))
#     crp.save("results/cropped.png")
    
    # Determine where to place the cropped image in wimg1
    # (Assume we place it at the same bounding box location)
    wimg1[y_min:y_max, x_min:x_max] = cropped_img

    return wimg1

def segment_and_replace(img, points, wimg1, face_indices=list(range(0, 27))):  
    # Convert points to a numpy array
    points = np.array(points)
    
    # Filter out landmarks that are very close to the margins
    margin = 2
    filtered_points = [point for point in points if margin < point[0] < img.shape[1] - margin and margin < point[1] < img.shape[0] - margin]
    filtered_points = np.array(filtered_points)
    
    # Define acceptable indices for filtered points
    acceptable_indices = set(face_indices)
    
    # Further filter the already filtered points based on acceptable indices
    final_filtered_points = [filtered_points[i] for i in range(len(filtered_points)) if i in acceptable_indices]
    final_filtered_points = np.array(final_filtered_points)
    
    # Create the convex mask
    mask = create_convex_mask(img.shape, final_filtered_points)
    cv2.imwrite("results/mask.png", mask)  # Save the mask image
    
    # Apply the mask and replace the region in wimg1 with the corresponding region from img
    replaced_img = apply_mask_and_replace(img, wimg1, mask)
#     Image.fromarray(cv2.cvtColor(np.uint8(replaced_img), cv2.COLOR_BGR2RGB)).save("results/replaced_img.png")

    return replaced_img

def create_convex_mask(img_shape, points):
    """
    Creates a binary convex mask based on the provided points.

    Parameters:
    img_shape (tuple): Shape of the image (height, width).
    points (array-like): Array of points used to create the mask.

    Returns:
    np.ndarray: Binary mask with the same shape as the image.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Create a black mask with the same height and width as the image
    points = np.array(points, dtype=np.int32)  # Ensure the points are in the correct format for cv2.fillPoly

    if len(points) > 0:
        hull = cv2.convexHull(points)  # Create the convex hull of the points
        cv2.fillConvexPoly(mask, hull, 255)  # Fill the area defined by the convex hull with white (255)

    return mask

def apply_mask_and_replace(src_img, dest_img, mask):
    """
    Applies the mask to the source image and replaces the corresponding region in the destination image.

    Parameters:
    src_img (np.ndarray): Source image.
    dest_img (np.ndarray): Destination image.
    mask (np.ndarray): Binary mask.

    Returns:
    np.ndarray: Image with the region replaced.
    """
    # Ensure the mask is binary
    mask = mask // 255
    
    # Create an inverse mask
    inverse_mask = 1 - mask
    
    # Apply the mask to the source and destination images
    src_region = cv2.bitwise_and(src_img, src_img, mask=mask)
    dest_region = cv2.bitwise_and(dest_img, dest_img, mask=inverse_mask)
    
    # Combine the regions
    combined = cv2.add(src_region, dest_region)
    
    return combined


def local_morph(img1, img2, points1, points2, tri_list, alpha, output_path):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    points = []

    for i in range(len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))
        
    morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)
    img_out_1 = img1.copy()
    img_out_2 = img2.copy()
    for i in range(len(tri_list)):
        x, y, z = int(tri_list[i][0]), int(tri_list[i][1]), int(tri_list[i][2])
        t1, t2, t = [points1[x], points1[y], points1[z]], [points2[x], points2[y], points2[z]], [points[x], points[y], points[z]]
        morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha, img_out_1, img_out_2, draw_triangles=False, context = "local")
        
    
    img_out = segment_and_replace(morphed_frame, points, img_out_1)
    
    res = Image.fromarray(cv2.cvtColor(np.uint8(img_out), cv2.COLOR_BGR2RGB))
    res.save(output_path, 'PNG')