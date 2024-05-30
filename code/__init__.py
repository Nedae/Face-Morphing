from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence, generate_weighted_image, local_morph

import subprocess
import argparse
import shutil
import os
import cv2

def doMorphing(img1, img2, duration, frame_rate, output):
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, output)
    
    
def doMorphingImage(img1, img2, alpha, output):
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    generate_weighted_image(img1, img2, points1, points2, tri, size, alpha, output)
    
def doLocalMorphing(img1, img2, alpha, output, adjust_skin_tone):
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    local_morph(img1, img2, points1, points2, tri, alpha, output, adjust_skin_tone)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="The First Image")
    parser.add_argument("--src", required=True, help="The Second Image")
    parser.add_argument("--duration", type=int, default=5, help="The duration")
    parser.add_argument("--frame", type=int, default=20, help="The frameame Rate")
    parser.add_argument("--output", required=True, help="Output file path (GIF or PNG/JPG)")
    parser.add_argument("--alpha", default=0.5, type=float, help="Alpha value for single image morphing (0 to 1)")
    parser.add_argument("--local", action="store_true", help="Perform local morphing")
    parser.add_argument("--adjust_skin_tone", action="store_true", help="Perform local morphing")

    args = parser.parse_args()

    image1 = cv2.imread(args.dest)
    image2 = cv2.imread(args.src)

    # Determine if the output is a GIF or a single image
    if args.output.lower().endswith('.gif'):
        assert not args.local, "Local morphing can only be done when the output is an image type (PNG/JPG)"
        doMorphing(image1, image2, args.duration, args.frame, args.output)
    else:
        if args.local:
            doLocalMorphing(image1, image2, args.alpha, args.output, args.adjust_skin_tone)
        else:
            doMorphingImage(image1, image2, args.alpha, args.output)
