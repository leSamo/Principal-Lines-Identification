# Identification of Persons According to Principal Lines on a Palm
# Biometric Systems 2023/24
# Samuel Olekšák

import os
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_SOURCE = "images/"

# Whole hand contour extraction
THRESHOLD_VALUE = 40
BORDER_THICKNESS = 10
WRIST_PORTION = 1 / 6

# ROI parameters
WIDTH_MODIFIER = 1.2
ROI_START_OFFSET = 30

# Feature extraction parameters
CONTOUR_COUNT = 3

class Side(Enum):
    LEFT = "l"
    RIGHT = "r"

class Wavelength(Enum):
    NM460 = "460"
    NM630 = "630"
    NM700 = "700"
    NM850 = "850"
    NM940 = "940"
    WHITE = "WHT"

# Defines a single image in the dataset along with its metadata
@dataclass
class Sample():
    person: int
    side: Side
    wavelength: Wavelength
    index: int
    image: np.ndarray
    features: np.ndarray

# Load input image defined by filename into numpy array,
# extract metadata from filename and convert into Sample object
def parse_sample(filepath):
    _, filename_with_extension = os.path.split(filepath)
    filename, _ = os.path.splitext(filename_with_extension)

    person, side, wavelength, index = filename.split("_")
    person = int(person)
    index = int(index)

    image = cv2.imread(filepath)

    return Sample(person, side, wavelength, index, image, [])

# Load all images from the source folder into a list of Sample objects
dataset_filepath = os.listdir(DATASET_SOURCE)
dataset = [parse_sample(DATASET_SOURCE + filepath) for filepath in dataset_filepath]
dataset = [sample for sample in dataset if sample.wavelength == "460" and sample.side == "l"]

print(f"Loaded {len(dataset)} images")

# ==================================================

def get_contours(original_image):
    # Remove right 1/6 of the image since it contains only wrist and distracting features
    height, width, channels = original_image.shape
    borderless_image = original_image[:, :int(width*(1 - WRIST_PORTION))]
    height, width, channels = borderless_image.shape

    borderless_image_original = borderless_image.copy()

    # Binary thresholding for distinguishing palm from the background
    _, borderless_image = cv2.threshold(borderless_image, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # problem: sometimes the tips of the fingers are exceeding the confines of the image
    # solution: set a few bottom, top and right pixel rows to black, so the contours will continue to the wrist
    # Create a new image with the black border
    image = np.zeros((height + 2 * BORDER_THICKNESS, width + 2 * BORDER_THICKNESS, 3), dtype=np.uint8)
    image[BORDER_THICKNESS:BORDER_THICKNESS + height, BORDER_THICKNESS:BORDER_THICKNESS + width] = borderless_image
    height, width, channels = image.shape

    # Apply dilation and Gaussian blur to remove jagged edges and help connect the contours into one
    kernel = np.ones((9, 9), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (9, 9), 0)

    # Detect hand contour
    canny = cv2.Canny(image, 10, 47)

    # Blur the canny image a bit to fix loosely connected contours
    canny = cv2.GaussianBlur(canny, (3, 3), 0)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# ==================================================

# Takes in list of contours and returns one with the largest arcLength
def get_longest_contour(contours):
    longest_contour = None
    longest_contour_length = 0

    for contour in contours:
        length = cv2.arcLength(contour, True)

        if length > longest_contour_length:
            longest_contour_length = length
            longest_contour = contour

    return longest_contour

# ==================================================

# Accepts longest contour and returns pointA and pointB
# pointA -- between ring finger and little finger
# pointb -- between index finger and middle finger
def get_points(longest_contour):
    hull = cv2.convexHull(longest_contour, returnPoints=False)
    hull[::-1].sort(axis=0)

    defects = cv2.convexityDefects(longest_contour, hull)

    points = []

    # Iterate through the defects and find significant ones
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 20000:
            points.append(tuple(longest_contour[f][0]))
    
    # Sort points by y value
    points.sort(key=lambda p: p[1])

    pointA = points[0] # between ring and little finger
    pointB = points[2] # between index and middle finger

    return pointA, pointB

# ==================================================

# accepts pointA and pointB and returns tuple consisting of four points defining
# ROI square region and of vector between pointB and pointA
def get_roi_params(pointA, pointB):
    midpoint = ((pointA[0] + pointB[0]) // 2, (pointA[1] + pointB[1]) // 2)
    vector = (pointB[0] - pointA[0], pointB[1] - pointA[1])
    perpendicular_vector = (vector[1], -vector[0])

    perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
    vector_length = np.linalg.norm(vector)

    normalized_perpendicular_vector = (perpendicular_vector[0] / perpendicular_vector_length, perpendicular_vector[1] / perpendicular_vector_length)
    normalized_vector = (vector[0] / vector_length, vector[1] / vector_length)

    roi_start = (int(midpoint[0] + normalized_perpendicular_vector[0] * ROI_START_OFFSET), int(midpoint[1] + normalized_perpendicular_vector[1] * ROI_START_OFFSET))

    roi_side_length_half = int(np.linalg.norm(perpendicular_vector) * WIDTH_MODIFIER / 2)

    vertices = np.array(
        [
            [int(roi_start[0] + normalized_vector[0] * roi_side_length_half), int(roi_start[1] + normalized_vector[1] * roi_side_length_half)],
            [int(roi_start[0] - normalized_vector[0] * roi_side_length_half), int(roi_start[1] - normalized_vector[1] * roi_side_length_half)],
            [int(roi_start[0] - normalized_vector[0] * roi_side_length_half + normalized_perpendicular_vector[0] * roi_side_length_half * 2), int(roi_start[1] - normalized_vector[1] * roi_side_length_half + normalized_perpendicular_vector[1] * roi_side_length_half * 2)],
            [int(roi_start[0] + normalized_vector[0] * roi_side_length_half + normalized_perpendicular_vector[0] * roi_side_length_half * 2), int(roi_start[1] + normalized_vector[1] * roi_side_length_half + normalized_perpendicular_vector[1] * roi_side_length_half * 2)],
        ], dtype=np.int32)

    vertices = vertices.reshape((-1, 1, 2))

    return vertices, vector

# ==================================================

# Receives vector between pointB and pointA and original image;
# returns rotated original image such that ROI is axis aligned and rotation matrix
def find_rotation(vector, image):
    angle_radians = np.arctan2(vector[1], vector[0])
    angle_degrees = np.degrees(angle_radians)

    height, width, channels = image.shape
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees - 180, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image, rotation_matrix

# ==================================================

# Extracts ROI from original image
# Returns histogram equalized and non-equalized images
def get_roi(original_image, vertices, rotation_matrix):
    rotated_vertices = cv2.transform(vertices, rotation_matrix)

    rotated_vertices = rotated_vertices.reshape(-1, 2)

    roi = original_image[rotated_vertices[0][1]:rotated_vertices[2][1], rotated_vertices[0][0]:rotated_vertices[1][0]]
    non_equalized = roi.copy()

    roi = cv2.equalizeHist(roi[:,:,0])
    roi = np.repeat(roi[:, :, np.newaxis], 3, axis=2)

    return roi, non_equalized

# ==================================================

# Performs filtration with 24 Gabor filters with various angles theta
def get_gabor(roi):
    angle_range = np.arange(1.4, 6.3, 0.2)
    images = []

    # Gabor filter parameters
    ksize = (31, 31) # Kernel size
    sigma = 0.8      # Standard deviation
    lambd = 25       # Wavelength
    psi = 0          # Phase offset

    for theta in angle_range:
        kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, psi)
        roi_blur = cv2.GaussianBlur(roi, (11, 11), 5)
        roi_gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.filter2D(roi_gray, -1, kernel)
        images.append(filtered_image)

    dst = np.average(images, axis=0)
    dst = np.uint8(dst)
    dst = cv2.resize(dst, (188 * 2, 187 * 2), interpolation=cv2.INTER_NEAREST)

    return dst

# ==================================================

# Returns tuple -- first value is a boolean signifying whether the extraction was successful
# seconds value is a 3x3 array of features for the image
def get_features(roi, index):
    def get_contours(image):
        clahe = cv2.createCLAHE(clipLimit=1)
        image = clahe.apply(image)

        adaptive_threshold = cv2.adaptiveThreshold(-image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -5)

        contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)

        return sorted_contours[:CONTOUR_COUNT]
    
    gabor = get_gabor(roi)

    gabor = (gabor - gabor.min()) / (gabor.max() - gabor.min()) * 255
    gabor = gabor.astype("uint8")

    contours1 = get_contours(gabor)

    if len(contours1) < CONTOUR_COUNT:
        return False, np.array([[0,0,0] for i in range(CONTOUR_COUNT)])

    contours1_moments = [cv2.moments(contour1) for contour1 in contours1]

    contours1_y = [x['m01'] / (x['m00'] + 1e-8) for x in contours1_moments]

    contours1 = sorted(zip(contours1_y, contours1), key=lambda x: x[0])
    _, contours1 = zip(*contours1)

    np.set_printoptions(suppress=True)

    c = []

    for contour in contours1:
        reshaped_array = contour.reshape((len(contour), 2))

        # Split the reshaped array into two separate arrays
        x = reshaped_array[:, 0]
        y = reshaped_array[:, 1]

        coefficients = np.polyfit(x, y, 2)
        
        c.append(coefficients)
    
    return True, c

# ==================================================

# Groups all previous functions into a pipeline to process original image into features
def process_image(original_image, index):
    contours = get_contours(original_image)
    longest_contour = get_longest_contour(contours)

    pointA, pointB = get_points(longest_contour)
    vertices, vector = get_roi_params(pointA, pointB)
    rotated_image, rotation_matrix = find_rotation(vector, original_image)
    roi, non_equalized = get_roi(rotated_image, vertices, rotation_matrix)
    found, features = get_features(roi, index)

    return found, features

# ==================================================

# Process the whole dataset
for index, sample in enumerate(dataset):
    sample.features = process_image(sample.image, index)

# ==================================================

# Split dataset into training and testing parts
training_dataset = [sample for sample in dataset if sample.index in [1,2,3]]
testing_dataset  = [sample for sample in dataset if sample.index in [4,5,6]]

# Find median values for all three coeffients
# These values will be used for normalization
normalizer = [[] for i in range(3)]

for sample in training_dataset:
    for contourIndex, contour in enumerate(sample.features[1]):
        for coefIndex, coef in enumerate(contour):
            normalizer[coefIndex].append(coef)

normalizer = np.array([np.median(np.abs(c)) for c in normalizer])

# Count the success rate
passed = 0
failed = 0
errored = 0

# Evaluate all testing images
for bindex, baseline in enumerate(testing_dataset):
    distances = []

    if baseline.features[0] == False:
        print("\033[38;5;208mFAILED: Could not extract data\033[0m")
        errored += 1
        continue

    for sindex, sample in enumerate(training_dataset):
        if sample.features[0] == False:
            distances.append(float('inf'))
        else:
            distances.append(np.mean([np.linalg.norm((c1 - c2) / normalizer, 2) for c1, c2 in zip(baseline.features[1], sample.features[1])]))

    if training_dataset[np.argmin(distances)].person == baseline.person:
        print(f"\033[32mPASS: Matched {bindex} to {np.argmin(distances)} (person {baseline.person}) with distance {np.min(distances):.3f}\033[0m")
        passed += 1
    else:
        print(f"\033[31mFAILED: Matched {bindex} (person {baseline.person}) to {np.argmin(distances)} (person {training_dataset[np.argmin(distances)].person}) with distance {np.min(distances):.3f}\033[0m")
        failed += 1

print(f"\033[33mPASSED {passed}/{passed + failed + errored} (accuracy {passed / (passed + failed + errored) * 100}%)\033[0m")
