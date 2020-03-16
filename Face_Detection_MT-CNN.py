# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:51:53 2020

@author: hp
"""

# Set working directory

import os
print("Current Working Directory " , os.getcwd())
#os.chdir("F:")

# Confirm MTCNN was installed correctly

import mtcnn

# Print version

print(mtcnn.__version__)

#           Face detection with mtcnn on a photograph

# Import required packages

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

# Draw an image with detected objects

def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# Plot the image
	pyplot.imshow(data)
	# Get the context for drawing boxes
	ax = pyplot.gca()
	# Plot each box
	for result in result_list:
		# Get coordinates
		x, y, width, height = result['box']
		# Create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# Draw the box
		ax.add_patch(rect)
		# Draw the dots
		for key, value in result['keypoints'].items():
			# Create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# Show the plot
	pyplot.show()

# Draw each face separately
def draw_faces(filename, result_list):
	# Load the image
	data = pyplot.imread(filename)
	# Plot each face as a subplot
	for i in range(len(result_list)):
		# Get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# Define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# Plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# Show the plot
	pyplot.show()

filename = 'FD_test2.jpg'

# Load image from file

pixels = pyplot.imread(filename)

# Create the detector, using default weights

detector = MTCNN()

# Detect faces in the image

faces = detector.detect_faces(pixels)

# Display faces on the original image

draw_image_with_boxes(filename, faces)

# Display faces on the original image
draw_faces(filename, faces)
