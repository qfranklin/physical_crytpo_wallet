import qrcode
import numpy as np
from stl import mesh
from PIL import Image
import math
import bpy

from os import system; 
cls = lambda: system('cls'); 
cls()

# Parameters
base_thickness = 0.6  # Thickness of the baseplate in mm
qr_thickness = 0.4  # Thickness of the QR code extrusion in mm
desired_size = 30.0  # Desired width and height of the QR code and baseplate in mm
box_size = 1.0  # Size of each box in the QR code (adjust this if needed)
grid_size = (4, 4)  # Customize grid size (rows, columns)
space_between_qrs = 5.0  # Space between QR codes in mm

# Generate QR Code and save as image
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=box_size,
    border=4,  # Border size in boxes
)
qr.add_data('Quentin Franklin 7573538453')
qr.make(fit=True)

# Save the QR code as an image
img = qr.make_image(fill='black', back_color='white')
img.save(r'C:\Users\qfran\Desktop\code\qr_code\qrcode.png')

# Convert QR code image to height map (3D mesh)
img = Image.open(r'C:\Users\qfran\Desktop\code\qr_code\qrcode.png').convert('L')  # Convert to grayscale
pixels = np.array(img)  # Get pixel data as a numpy array
height, width = pixels.shape

# Calculate scaling factors
x_scale = desired_size / width
y_scale = desired_size / height
z_scale = qr_thickness  # Extrusion height

# Prepare vertices and faces for STL
vertices = []
faces = []

# Define function to shift vertices based on QR code's position in the grid
def shift_vertices(v_list, x_shift, y_shift):
    return [[x + x_shift, y + y_shift, z] for x, y, z in v_list]

# Loop over the grid to place multiple QR codes
for row in range(grid_size[0]):
    for col in range(grid_size[1]):
        x_offset = col * (width * x_scale + space_between_qrs)
        y_offset = row * (height * y_scale + space_between_qrs)
        
        # Define baseplate vertices for each QR code
        base_vertices = [
            [0, 0, 0],
            [width * x_scale, 0, 0],
            [width * x_scale, height * y_scale, 0],
            [0, height * y_scale, 0],
            [0, 0, base_thickness],
            [width * x_scale, 0, base_thickness],
            [width * x_scale, height * y_scale, base_thickness],
            [0, height * y_scale, base_thickness]
        ]
        
        # Shift the baseplate vertices according to the grid position
        base_vertices = shift_vertices(base_vertices, x_offset, y_offset)
        idx = len(vertices)
        
        # Add baseplate vertices and faces
        vertices.extend(base_vertices)
        faces.extend([
            [idx, idx + 1, idx + 2], [idx, idx + 2, idx + 3], 
            [idx + 4, idx + 5, idx + 6], [idx + 4, idx + 6, idx + 7],
            [idx, idx + 1, idx + 5], [idx, idx + 5, idx + 4],
            [idx + 1, idx + 2, idx + 6], [idx + 1, idx + 6, idx + 5],
            [idx + 2, idx + 3, idx + 7], [idx + 2, idx + 7, idx + 6],
            [idx + 3, idx, idx + 4], [idx + 3, idx + 4, idx + 7]
        ])
        
        # Define QR code vertices and faces for each pixel
        for y in range(height):
            for x in range(width):
                if pixels[y, x] < 128:  # Black pixels only (for QR code)
                    z = z_scale  # Set height for black pixels
                else:
                    z = 0  # Set flat for white pixels

                # Define 4 vertices of the cube for the pixel
                v1 = [x * x_scale, y * y_scale, base_thickness]
                v2 = [(x + 1) * x_scale, y * y_scale, base_thickness]
                v3 = [(x + 1) * x_scale, (y + 1) * y_scale, base_thickness]
                v4 = [x * x_scale, (y + 1) * y_scale, base_thickness]
                v5 = [x * x_scale, y * y_scale, base_thickness + z]
                v6 = [(x + 1) * x_scale, y * y_scale, base_thickness + z]
                v7 = [(x + 1) * x_scale, (y + 1) * y_scale, base_thickness + z]
                v8 = [x * x_scale, (y + 1) * y_scale, base_thickness + z]

                # Shift QR code pixels for grid positioning
                v1, v2, v3, v4, v5, v6, v7, v8 = shift_vertices([v1, v2, v3, v4, v5, v6, v7, v8], x_offset, y_offset)
                
                idx = len(vertices)
                vertices.extend([v1, v2, v3, v4, v5, v6, v7, v8])

                # Create faces for the cube (6 faces per cube)
                faces.extend([
                    [idx, idx + 1, idx + 5], [idx, idx + 5, idx + 4],
                    [idx + 1, idx + 2, idx + 6], [idx + 1, idx + 6, idx + 5],
                    [idx + 2, idx + 3, idx + 7], [idx + 2, idx + 7, idx + 6],
                    [idx + 3, idx, idx + 4], [idx + 3, idx + 4, idx + 7],
                    [idx + 4, idx + 5, idx + 6], [idx + 4, idx + 6, idx + 7]
                ])

# Convert lists to numpy arrays for STL creation
vertices = np.array(vertices)
faces = np.array(faces)

# Create STL mesh and save
qr_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(faces):
    for j in range(3):
        qr_mesh.vectors[i][j] = vertices[face[j], :]

# Save as STL
qr_mesh.save('qr_code_grid.stl')