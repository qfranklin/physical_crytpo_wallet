import qrcode
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

from scipy.spatial.transform import Rotation as R

import config.config as config

def rotate_stl(stl_file, angle_deg=270):
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_file)

    # Convert the angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # Rotation matrix for Z-axis rotation
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                1]
    ])

    # Apply the rotation to each point in the mesh
    your_mesh.vectors = np.dot(your_mesh.vectors, rotation_matrix)

    # Save the rotated STL
    your_mesh.save(stl_file)

def add_vertices(vertices, x_offset, y_offset, x_scale, y_scale, thickness, z, width, height):
    """
    Adds vertices to create a cube.
    
    Arguments:
    - vertices: list to which the new vertices will be added.
    - x_offset: the X-coordinate offset for positioning.
    - y_offset: the Y-coordinate offset for positioning.
    - x_scale: scaling factor for the X dimension.
    - y_scale: scaling factor for the Y dimension.
    - thickness: thickness of the object.
    - z: the additional height (thickness) on top of the base (default is 0 for a flat plate).
    - width: the width in terms of the number of grid units for the object (default is 1).
    - height: the height in terms of the number of grid units for the object (default is 1).
    """
    vertices.extend([
        [x_offset, y_offset, thickness],  # Bottom-left
        [x_offset + width * x_scale, y_offset, thickness],  # Bottom-right
        [x_offset + width * x_scale, y_offset + height * y_scale, thickness],  # Top-right
        [x_offset, y_offset + height * y_scale, thickness],  # Top-left
        [x_offset, y_offset, thickness + z],  # Bottom-left (raised)
        [x_offset + width * x_scale, y_offset, thickness + z],  # Bottom-right (raised)
        [x_offset + width * x_scale, y_offset + height * y_scale, thickness + z],  # Top-right (raised)
        [x_offset, y_offset + height * y_scale, thickness + z]  # Top-left (raised)
    ])

def add_faces(faces, start_idx):
    """
    Adds the faces for a cube (or rectangular prism) to the face list.

    Arguments:
    - faces: list to which the new faces will be added.
    - start_idx: the index at which the cube's vertices start in the vertices list.
    """
    faces.extend([
        [start_idx, start_idx + 1, start_idx + 2], [start_idx, start_idx + 2, start_idx + 3],  # Bottom face
        [start_idx + 4, start_idx + 5, start_idx + 6], [start_idx + 4, start_idx + 6, start_idx + 7],  # Top face
        [start_idx, start_idx + 1, start_idx + 5], [start_idx, start_idx + 5, start_idx + 4],  # Side face
        [start_idx + 1, start_idx + 2, start_idx + 6], [start_idx + 1, start_idx + 6, start_idx + 5],  # Side face
        [start_idx + 2, start_idx + 3, start_idx + 7], [start_idx + 2, start_idx + 7, start_idx + 6],  # Side face
        [start_idx + 3, start_idx, start_idx + 4], [start_idx + 3, start_idx + 4, start_idx + 7]  # Side face
    ])

def import_sd_card():
    # Load the SD card STL file once outside the loop
    sd_card_mesh = mesh.Mesh.from_file(config.current_directory + 'sd_card_bottom.stl')
    sd_card_vertices = sd_card_mesh.vectors.reshape(-1, 3)  # Extract vertices into a NumPy array
    sd_card_faces = np.arange(len(sd_card_vertices)).reshape(-1, 3)  # Generate face indices

    # Rotate the sd card
    rotation_angle = np.radians(270)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])
    sd_card_vertices = sd_card_vertices @ rotation_matrix.T

    # Move the sd card so the bottom left is aligned with the top left of the qr code and the z axis is zeroed and level
    sd_card_vertices[:, 0] -= np.max(sd_card_vertices[:, 0])  # Align bottom left corner with (0, 0)
    sd_card_vertices[:, 1] -= np.min(sd_card_vertices[:, 1])  # Level the bottom of the SD card
    sd_card_vertices[:, 2] -= np.min(sd_card_vertices[:, 2])  # Zero out the Z axis

    # Calculate width and height
    sd_card_width = np.max(sd_card_vertices[:, 0]) - np.min(sd_card_vertices[:, 0])
    sd_card_height = np.max(sd_card_vertices[:, 1]) - np.min(sd_card_vertices[:, 1])

    return sd_card_vertices, sd_card_faces, sd_card_width, sd_card_height

def generate_qr_code(vertices, faces, pixels, size, protrusion_thickness, base_thickness, x_offset, y_offset):

    height, width = pixels.shape

    # Calculate scaling factors
    x_scale = size / width
    y_scale = size / height

    # Add baseplate vertices and faces
    # Add vertices for baseplate
    add_vertices(vertices, x_offset, y_offset, x_scale, y_scale, 0, base_thickness, width, height)
    add_faces(faces, 0)

    # Define QR code vertices and faces for each pixel
    for y in range(height):
        for x in range(width):
            if pixels[y, x] < 128 or x == 0 or y == 0 or (x + 1) == width or (y + 1) == height:  # Black pixels only (for QR code)
                z = protrusion_thickness  # Set height for black pixels
            else:
                z = 0  # Set flat for white pixels

            qr_idx = len(vertices)

            # Add vertices for each QR cube
            add_vertices(vertices, x * x_scale + x_offset, y * y_scale + y_offset, x_scale, y_scale, base_thickness, z, 1, 1)
            add_faces(faces, qr_idx)

def import_qr_code(text):
    # Generate QR Code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=5, # 4 units is the qr code standard, but add one to account for outline
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white')
    img = img.convert('L')  # Convert to grayscale
    
    return np.array(img)  # Get pixel qr_code_text as a numpy array
    
def qr_code():

    # These variables are in milimeters
    desired_size = 45
    layer_height = 0.16
    protrusion_thickness = layer_height * 2
    base_thickness = layer_height * 5
    base_extension = 14
    space_between_qrs = 5

    sd_card_vertices, sd_card_faces, sd_card_width, sd_card_height = import_sd_card()

    all_vertices = []
    all_faces = []

    # Calculate grid layout dynamically
    total_qr_codes = len(config.qr_code_text)
    grid_size = math.ceil(math.sqrt(total_qr_codes))

    # Loop over the list of public/private keys
    for idx, qr_code_text in enumerate(config.qr_code_text_array):
        # Determine row and column position for each QR code
        col = idx // grid_size
        row = idx % grid_size
        x_offset = col * (desired_size + base_extension + (space_between_qrs * idx))
        y_offset = row * (desired_size + (space_between_qrs * idx))

        vertices = []
        faces = []

        pixels = import_qr_code(qr_code_text)
        generate_qr_code(vertices, faces, pixels, desired_size, protrusion_thickness, base_extension, x_offset, y_offset)

        height, width = pixels.shape

        # Calculate scaling factors
        x_scale = desired_size / width
        y_scale = desired_size / height
        z_scale = protrusion_thickness  # Extrusion height

        # Prepare vertices and faces for STL
        vertices = []
        faces = []

        # Add baseplate vertices and faces
        # Add vertices for baseplate
        add_vertices(vertices, x_offset, y_offset, x_scale, y_scale, 0, base_thickness, width, height)
        add_faces(faces, 0)

        # Define QR code vertices and faces for each pixel
        for y in range(height):
            for x in range(width):
                if pixels[y, x] < 128 or x == 0 or y == 0 or (x + 1) == width or (y + 1) == height:  # Black pixels only (for QR code)
                    z = z_scale  # Set height for black pixels
                else:
                    z = 0  # Set flat for white pixels

                qr_idx = len(vertices)

                # Add vertices for each QR cube
                add_vertices(vertices, x * x_scale + x_offset, y * y_scale + y_offset, x_scale, y_scale, base_thickness, z, 1, 1)
                add_faces(faces, qr_idx)

        # Add SD card model to this QR code
        sd_offset_x = x_offset  # Adjust as needed for correct positioning
        sd_offset_y = y_offset
        sd_offset_z = 0 #base_thickness + protrusion_thickness  # Place on top of the baseplate

        # Adjust the SD card vertices for its new position
        sd_card_vertices_adjusted = sd_card_vertices.copy()
        sd_card_vertices_adjusted[:, 0] += sd_offset_x  # X offset
        sd_card_vertices_adjusted[:, 1] += sd_offset_y  # Y offset
        sd_card_vertices_adjusted[:, 2] += sd_offset_z  # Z offset

        current_vertex_offset = len(vertices)
        vertices.extend(sd_card_vertices_adjusted.tolist())

        # Adjust the SD card faces and add them to the QR code faces
        for face in sd_card_faces:
            adjusted_face = [f + current_vertex_offset for f in face]
            faces.append(adjusted_face)

        extension_width = int(round(desired_size / x_scale, 0))
        extension_height = int(round(base_extension / y_scale, 0))
        adjacency_range = extension_height

        # Add the base extension
        for y in range(extension_width):
            for x in range(extension_height):

                if y == 0 or \
                  (x + 1) == extension_height or \
                  (y + 1) == extension_width or \
                  (((extension_height - 1 - x) + (extension_width - 1 - y)) == adjacency_range - 1):
                    z = z_scale
                else:
                    z = 0

                if ((extension_height - 1 - x) + (extension_width - 1 - y)) < adjacency_range - 1:
                    continue

                qr_idx = len(vertices)

                if ((extension_height - 1 - x) + (extension_width - 1 - y)) == adjacency_range:
                    # This will make the edge cubes have a 45 degreee edge.
                    vertices.extend([
                        [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                        [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, 0],
                        [(x + .5) * x_scale + desired_size, (y + .5) * y_scale + y_offset, 0],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + .5) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness + z],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                    ])
                elif ((extension_height - 1 - x) + (extension_width - 1 - y)) == adjacency_range - 1:

                    if (y + 1) == extension_width:
                        vertices.extend([
                            [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                    elif (x + 1) == extension_height: 
                        vertices.extend([
                            [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, (y + .5) * y_scale + y_offset, 0],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, base_thickness + z],
                            [(x + 1) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness + z],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                    else:
                        vertices.extend([
                            [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                else:
                    add_vertices(vertices, x * x_scale + desired_size, y * y_scale + y_offset, x_scale, y_scale, 0, base_thickness + z, 1, 1)
        
                add_faces(faces, qr_idx)

        # Next section is for adding text to the bottom of the qr code. 
        font_size = 11
        font = ImageFont.truetype(config.current_directory + "text_font.ttf", font_size)

        text_width = 500
        text_height = 500
        text_image = Image.new('L', (text_width, text_height), color=255)
        
        text_draw = ImageDraw.Draw(text_image)

        text_x_position = (desired_size * idx) + (space_between_qrs * idx) + 1
        text_y_position = desired_size + 1

        # Draw the text on the new larger image
        text_draw.text((text_x_position, text_y_position), config.front_side_text[idx], fill=0, font=font)

        # Correctly orient the image
        text_image = ImageOps.mirror(text_image)
        text_image = text_image.rotate(90, expand=True)

        # Convert this text image to 3D vertices (black pixels = protruding)
        text_pixels = text_image.load()

        # Loop through the pixels in the text image and generate vertices
        for y in range(text_image.height):
            for x in range(text_image.width):
                if text_pixels[x, y] < 128:  # Black pixels = protruding areas
                    text_idx = len(vertices)

                    add_vertices(vertices, x, y, 1, 1, base_thickness, z_scale, 1, 1)
                    add_faces(faces, text_idx)

        current_vertex_offset = len(all_vertices)
        all_vertices.extend(vertices)
        all_faces.extend([[f[0] + current_vertex_offset, f[1] + current_vertex_offset, f[2] + current_vertex_offset] for f in faces])

    # Convert lists to numpy arrays for STL creation
    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)

    # Create STL mesh and save
    qr_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            qr_mesh.vectors[i][j] = all_vertices[face[j], :]

    stl_file = config.current_directory + 'qr.stl'
    qr_mesh.save(rf'{stl_file}')

    #rotate_stl(stl_file, 90)

    return round(base_thickness + layer_height, 2)


def load_and_transform_stl(stl_file, translation=(0, 0, 0), rotation_degrees=(0, 0, 0)):
    # Load the STL
    sd_card_mesh = mesh.Mesh.from_file(stl_file)

    # Apply translation
    sd_card_mesh.translate(translation)

    # Apply rotation
    r = R.from_euler('xyz', rotation_degrees, degrees=True)
    sd_card_mesh.rotate_using_matrix(r.as_matrix())

    return sd_card_mesh

def combine_stl(main_mesh, new_mesh):
    # Combine the vertices of both meshes
    combined_vectors = np.concatenate([main_mesh.vectors, new_mesh.vectors], axis=0)
    
    # Create the combined mesh with the combined vectors
    combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, vector in enumerate(combined_vectors):
        combined_mesh.vectors[i] = vector

    return combined_mesh