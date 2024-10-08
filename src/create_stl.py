import qrcode
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

from scipy.spatial.transform import Rotation as R

import config.config as config

# These variables are in milimeters
desired_size = 40
layer_height = 0.16
protrusion_thickness = layer_height * 2
base_thickness = layer_height * 5
space_between_qrs = 5
all_vertices = []
all_faces = []

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
    rotation_angle = np.radians(180)
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

def generate_qr_code(vertices, faces, pixels, x_offset, y_offset):

    height, width = pixels.shape

    x_scale = desired_size / width
    y_scale = desired_size / height

    for y in range(height):
        for x in range(width):
            if pixels[y, x] < 128:
                z = protrusion_thickness
            else:
                z = 0

            idx = len(vertices)

            vertices.extend([
                [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset, base_thickness],
                [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset + 1 * y_scale, base_thickness],
                [x * x_scale + x_offset, y * y_scale + y_offset + 1 * y_scale, base_thickness],
                [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness + z],
                [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset, base_thickness + z],
                [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset + 1 * y_scale, base_thickness + z],
                [x * x_scale + x_offset, y * y_scale + y_offset + 1 * y_scale, base_thickness + z]
            ])

            add_faces(faces, idx)

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
    
def generate_sd_card(vertices, faces, sd_card_vertices, sd_card_faces, x_offset, y_offset):
    
     # Adjust the SD card vertices for its new position
    sd_card_vertices_adjusted = sd_card_vertices.copy()
    sd_card_vertices_adjusted[:, 0] += x_offset
    sd_card_vertices_adjusted[:, 1] += y_offset
    sd_card_vertices_adjusted[:, 2] += 0

    current_vertex_offset = len(vertices)
    vertices.extend(sd_card_vertices_adjusted.tolist())

    # Adjust the SD card faces and add them to the QR code faces
    for face in sd_card_faces:
        adjusted_face = [f + current_vertex_offset for f in face]
        faces.append(adjusted_face)

def generate_base(vertices, faces, width, height, x_offset, y_offset):

    idx = len(vertices)
    vertices.extend([
        [x_offset, y_offset, 0],
        [x_offset + height, y_offset, 0],
        [x_offset + height, y_offset + width, 0],
        [x_offset, y_offset + width, 0],
        [x_offset, y_offset, base_thickness],
        [x_offset + height, y_offset, base_thickness],
        [x_offset + height, y_offset + width, base_thickness],
        [x_offset, y_offset + width, base_thickness]
    ])
    add_faces(faces, idx)

def generate_outline(vertices, faces, sides, width, height, x_scale, y_scale, x_offset, y_offset):

    for y in range(width):
        for x in range(height):
            # Check for sides and add outline accordingly
            add_outline = False

            if sides[0] == 1 and x == 0:  # Top side
                add_outline = True
            elif sides[1] == 1 and (y + 1) == width:  # Right side
                add_outline = True
            elif sides[2] == 1 and (x + 1) == height:  # Bottom side
                add_outline = True
            elif sides[3] == 1 and y == 0:  # Left side
                add_outline = True

            if add_outline:
                idx = len(vertices)

                # Add the vertices for the outline
                vertices.extend([
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness + layer_height * 3],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, base_thickness + layer_height * 3],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + layer_height * 3],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + layer_height * 3]
                ])

                # Add the faces for this outline section
                add_faces(faces, idx)

def generate_angled_base(vertices, faces, width, height, x_scale, y_scale, x_offset, y_offset):
    
    width = int(round(width / x_scale, 0))
    height = int(round(height / y_scale, 0))

    # Set this to height and set the width to desired_size if not placing sd card in baseplate
    adjacency_range = -2 #height

    # Add the base extension
    for y in range(width):
        for x in range(height):

            if y == 0 or \
                (x + 1) == height or \
                (y + 1) == width or \
                (((height - 1 - x) + (width - 1 - y)) == adjacency_range - 1):
                z = 0
            else:
                z = 0

            if ((height - 1 - x) + (width - 1 - y)) < adjacency_range - 1:
                continue

            qr_idx = len(vertices)

            if ((height - 1 - x) + (width - 1 - y)) == adjacency_range:
                # This will make the edge cubes have a 45 degreee edge.
                vertices.extend([
                    [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                    [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                    [(x + .5) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness],
                    [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                    [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                    [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                    [(x + .5) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness + z],
                    [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                ])
            elif ((height - 1 - x) + (width - 1 - y)) == adjacency_range - 1:
                if (y + 1) == width:
                    vertices.extend([
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                        [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                    ])
                elif (x + 1) == height: 
                    vertices.extend([
                        [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, base_thickness],
                        [(x + 1) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, base_thickness + z],
                        [(x + 1) * x_scale + desired_size, (y + .5) * y_scale + y_offset, base_thickness + z],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                        [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                    ])
                else:
                    vertices.extend([
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                        [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness],
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + 1.5) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + .5) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                        [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                    ])
            else:
                vertices.extend([
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset + 1 * y_scale, base_thickness],
                    [x * x_scale + x_offset, y * y_scale + y_offset + 1 * y_scale, base_thickness],
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness + z],
                    [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset, base_thickness + z],
                    [x * x_scale + x_offset + 1 * x_scale, y * y_scale + y_offset + 1 * y_scale, base_thickness + z],
                    [x * x_scale + x_offset, y * y_scale + y_offset + 1 * y_scale, base_thickness + z]
                ])
            add_faces(faces, qr_idx)

def generate_text(vertices, faces, text, font, font_size, x_scale, y_scale, x_offset, y_offset):
    
    font = ImageFont.truetype(config.current_directory + font, font_size)

    text_image = Image.new('L', (500, 500), color=255)
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((x_offset, y_offset), text, fill=0, font=font)

    text_image = ImageOps.mirror(text_image)
    text_image = text_image.rotate(90, expand=True)

    text_pixels = text_image.load()

    for y in range(text_image.height):
        for x in range(text_image.width):
            if text_pixels[x, y] < 128:
                text_idx = len(vertices)

                vertices.extend([
                    [x * x_scale, y * y_scale, base_thickness],
                    [(x + 1) * x_scale, y * y_scale, base_thickness],
                    [(x + 1) * x_scale, (y + 1) * y_scale, base_thickness],
                    [x * x_scale, (y + 1) * y_scale, base_thickness],
                    [x * x_scale, y * y_scale, base_thickness + protrusion_thickness],
                    [(x + 1) * x_scale, y * y_scale, base_thickness + protrusion_thickness],
                    [(x + 1) * x_scale, (y + 1) * y_scale, base_thickness + protrusion_thickness],
                    [x * x_scale, (y + 1) * y_scale, base_thickness + protrusion_thickness]
                ])

                add_faces(faces, text_idx)

def qr_code():

    sd_card_vertices, sd_card_faces, sd_card_height, sd_card_width = import_sd_card()

    total_qr_codes = len(config.qr_code_text_array)
    grid_size = math.ceil(math.sqrt(total_qr_codes))

    for idx, qr_code_text in enumerate(config.qr_code_text_array):
        
        col = idx // grid_size
        row = idx % grid_size
        x_offset = col * (desired_size + sd_card_height + (space_between_qrs * idx))
        y_offset = row * (desired_size + (space_between_qrs * idx))

        vertices = []
        faces = []

        qr_code_x_offset = x_offset + 10
        pixels = import_qr_code(qr_code_text)
        generate_qr_code(vertices, faces, pixels, qr_code_x_offset, y_offset)

        height, width = pixels.shape
        x_scale = desired_size / width
        y_scale = desired_size / height

        generate_base(vertices, faces, desired_size, desired_size, qr_code_x_offset, y_offset)
        generate_outline(vertices, faces, [1,1,1,1], width, height, x_scale, y_scale, qr_code_x_offset, y_offset)

        #sd_card_x_offset = x_offset + sd_card_height + 10
        #sd_card_y_offset = y_offset + desired_size
        #generate_sd_card(vertices, faces, sd_card_vertices, sd_card_faces, sd_card_x_offset, sd_card_y_offset)

        baseplate_width = sd_card_width
        baseplate_height = desired_size
        baseplate_x_offset = 10
        baseplate_y_offset = desired_size
        baseplate_y_scale = baseplate_width / round(baseplate_width)
        baseplate_x_scale = baseplate_height / round(baseplate_height)
        generate_base(vertices, faces, baseplate_width, baseplate_height, baseplate_x_offset, baseplate_y_offset)
        generate_outline(vertices, faces, [1,1,1,0], round(baseplate_width), round(baseplate_height), baseplate_x_scale, baseplate_y_scale, baseplate_x_offset, baseplate_y_offset)

        text = config.front_right_text[idx]
        font = "8bitoperator_jve.ttf"
        font_size = 16
        text_x_scale = .36
        text_y_scale = 1.64
        text_x_position = (baseplate_y_offset + 2.2) / text_y_scale
        text_y_position = (baseplate_x_offset + 2) / text_x_scale
        generate_text(vertices, faces, text, font, font_size, text_x_scale, text_y_scale, text_x_position, text_y_position)

        baseplate_width = desired_size + sd_card_width
        baseplate_height = 10
        baseplate_x_offset = 0
        baseplate_y_offset = y_offset
        baseplate_y_scale = baseplate_width / round(baseplate_width)
        baseplate_x_scale = baseplate_height / round(baseplate_height)
        generate_base(vertices, faces, baseplate_width, baseplate_height, baseplate_x_offset, baseplate_y_offset)
        generate_outline(vertices, faces, [1,1,0,1], round(baseplate_width), round(baseplate_height), baseplate_x_scale, baseplate_y_scale, baseplate_x_offset, baseplate_y_offset)

        text = config.front_top_text[idx]
        font = "MinecraftBold.otf"
        font_size = 18
        text_x_scale = .45
        text_y_scale = .415
        text_x_position = (baseplate_y_offset + 3) / text_y_scale
        text_y_position = (baseplate_x_offset + 2) / text_x_scale
        generate_text(vertices, faces, text, font, font_size, text_x_scale, text_y_scale, text_x_position, text_y_position)


        print(f"{desired_size + sd_card_width}")

        current_vertex_offset = len(all_vertices)
        all_vertices.extend(vertices)
        all_faces.extend([[f[0] + current_vertex_offset, f[1] + current_vertex_offset, f[2] + current_vertex_offset] for f in faces])

    # Convert lists to numpy arrays for STL creation
    converted_vertices = np.array(all_vertices)
    converted_faces = np.array(all_faces)

    # Create STL mesh and save
    qr_mesh = mesh.Mesh(np.zeros(converted_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(converted_faces):
        for j in range(3):
            qr_mesh.vectors[i][j] = converted_vertices[face[j], :]

    stl_file = config.current_directory + 'qr.stl'
    qr_mesh.save(rf'{stl_file}')

    return round(base_thickness + layer_height, 2)
    # Combine the vertices of both meshes
    combined_vectors = np.concatenate([main_mesh.vectors, new_mesh.vectors], axis=0)
    
    # Create the combined mesh with the combined vectors
    combined_mesh = mesh.Mesh(np.zeros(combined_vectors.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, vector in enumerate(combined_vectors):
        combined_mesh.vectors[i] = vector

    return combined_mesh