import qrcode
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

from scipy.spatial.transform import Rotation as R

import config.config as config

# These variables are in milimeters
desired_size = 40
layer_height = 0.12
protrusion_thickness = layer_height * 2
base_thickness = layer_height * 4
space_between_qrs = 5
top_text_baseplate_width = desired_size / 3
top_text_baseplate_height = desired_size / 2
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

def import_qr_code(text, logo_scale=0.2, circle_radius=5, logo_text="Q"):
    # Generate QR Code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=1,  
        border=1
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white').convert('RGB')

    # Calculate the width and height of the QR code
    qr_width, qr_height = img.size

    # Calculate the center for the circular area
    center = (qr_width // 2, qr_height // 2)

    # Create a mask to cut out a circular area
    draw_qr = ImageDraw.Draw(img)

    # Clear the QR code area where the circular blank will be placed (set to white)
    draw_qr.ellipse(
        [center[0] - circle_radius, center[1] - circle_radius,
         center[0] + circle_radius, center[1] + circle_radius],
        fill='white'  # Fill the circle with white
    )

    img.save(config.current_directory + "qr_with_blank_circle.png")  # Save the resulting image

    # Convert the QR image to grayscale
    img = img.convert('L')

    # Convert image to numpy array
    pixels = np.array(img)

    # Flip the QR code to correct orientation
    pixels = np.flipud(pixels)

    return pixels

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
    
def generate_base(vertices, faces, thickness, width, height, x_offset, y_offset):

    idx = len(vertices)
    vertices.extend([
        [x_offset, y_offset, 0],
        [x_offset + height, y_offset, 0],
        [x_offset + height, y_offset + width, 0],
        [x_offset, y_offset + width, 0],
        [x_offset, y_offset, thickness],
        [x_offset + height, y_offset, thickness],
        [x_offset + height, y_offset + width, thickness],
        [x_offset, y_offset + width, thickness]
    ])
    add_faces(faces, idx)

def generate_outline(vertices, faces, sides, thickness, layers, width, height, x_scale, y_scale, x_offset, y_offset):

    for y in range(width):
        for x in range(height):
            # Check for sides and add outline accordingly
            add_outline = False

            if sides[0] == 1 and x < thickness:  # Top side
                add_outline = True
            elif sides[1] == 1 and (y + thickness) >= width:  # Right side
                add_outline = True
            elif sides[2] == 1 and (x + thickness) >= height:  # Bottom side
                add_outline = True
            elif sides[3] == 1 and y < thickness:  # Left side
                add_outline = True

            if add_outline:
                idx = len(vertices)

                # Add the vertices for the outline
                vertices.extend([
                    [x * x_scale + x_offset, y * y_scale + y_offset, 0],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, 0],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, 0],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, 0],
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness + layer_height * layers],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, base_thickness + layer_height * layers],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + layer_height * layers],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + layer_height * layers]
                ])

                # Add the faces for this outline section
                add_faces(faces, idx)

def generate_text(vertices, faces, text, font, font_size, x_size, y_size, x_scale, y_scale, x_offset, y_offset):
    
    font = ImageFont.truetype(config.current_directory + font, font_size)

    text_image = Image.new('L', (x_size, y_size), color=255)
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((x_offset, y_offset), text, fill=0, font=font)

    text_image = ImageOps.mirror(text_image)
    text_image = text_image.rotate(180, expand=True)

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

def generate_logo(vertices, faces, logo_path, logo_width, logo_height, protrusion_thickness, xy_offset):
    # Load the PNG logo
    logo_img = Image.open(logo_path).convert("L")  # Convert to grayscale
    logo_img = ImageOps.mirror(logo_img)
    logo_img = logo_img.rotate(180, expand=True)

    current_size = logo_img.size
    print(f"current size: {current_size}")

    # Convert image to a binary mask (1 for pixel, 0 for no pixel)
    logo_array = np.array(logo_img) > 128  # Adjust the threshold if necessary

    # Create a flat plane for the logo based on the image dimensions
    for y in range(logo_img.height - 1):
        for x in range(logo_img.width - 1):
            if logo_array[y, x]:  # If the pixel is part of the logo
                logo_idx = len(vertices)

                # Scale the pixel positions to fit within the logo_width and logo_height
                x0 = xy_offset[0] + x * (logo_width / logo_img.width)
                y0 = xy_offset[1] + y * (logo_height / logo_img.height)

                # Create the vertices for the base and protrusion
                vertices.extend([
                    [x0, y0, base_thickness],  # Base level
                    [x0 + (logo_width / logo_img.width), y0, base_thickness],
                    [x0 + (logo_width / logo_img.width), y0 + (logo_height / logo_img.height), base_thickness],
                    [x0, y0 + (logo_height / logo_img.height), base_thickness],
                    [x0, y0, base_thickness + protrusion_thickness],  # Protruding level
                    [x0 + (logo_width / logo_img.width), y0, base_thickness + protrusion_thickness],
                    [x0 + (logo_width / logo_img.width), y0 + (logo_height / logo_img.height), base_thickness + protrusion_thickness],
                    [x0, y0 + (logo_height / logo_img.height), base_thickness + protrusion_thickness]
                ])

                # Add the faces (two triangles for each of the base and protrusion faces)
                add_faces(faces, logo_idx)

def qr_code():

    sd_card_vertices, sd_card_faces, sd_card_height, sd_card_width = import_sd_card()

    total_qr_codes = len(config.qr_code_text_array)
    grid_size = math.ceil(math.sqrt(total_qr_codes))

    for idx, qr_code_text in enumerate(config.qr_code_text_array):
        
        col = idx // grid_size
        row = idx % grid_size
        x_offset = col * (desired_size + sd_card_height + (space_between_qrs * idx))
        y_offset = row * (desired_size + (space_between_qrs * idx) + top_text_baseplate_height)

        vertices = []
        faces = []

        #'''
        qr_code_x_offset = x_offset
        pixels = import_qr_code(qr_code_text)
        generate_qr_code(vertices, faces, pixels, qr_code_x_offset, y_offset)
        font_size = 80
        text_x_scale = .1
        text_y_scale = .1
        text_x_position = (22.5) / text_y_scale
        text_y_position = (40) / text_x_scale
        #generate_text(vertices, faces, "Q", "SuperMagic.ttf", font_size, text_x_scale, text_y_scale, text_x_position, text_y_position)

        height, width = pixels.shape
        x_scale = desired_size / width
        y_scale = desired_size / height

        generate_base(vertices, faces, base_thickness, desired_size, desired_size, qr_code_x_offset, y_offset)
        #generate_outline(vertices, faces, [1,1,1,1], 1, 2, width, height, x_scale, y_scale, qr_code_x_offset, y_offset)

        #sd_card_x_offset = x_offset + sd_card_height + 10
        #sd_card_y_offset = y_offset + desired_size
        #generate_sd_card(vertices, faces, sd_card_vertices, sd_card_faces, sd_card_x_offset, sd_card_y_offset)

        baseplate_x_offset = qr_code_x_offset + desired_size - top_text_baseplate_height
        baseplate_y_offset = y_offset + desired_size
        baseplate_y_scale = top_text_baseplate_width / round(top_text_baseplate_width)
        baseplate_x_scale = top_text_baseplate_height / round(top_text_baseplate_height)
        generate_base(vertices, faces, base_thickness, top_text_baseplate_width, top_text_baseplate_height, baseplate_x_offset, baseplate_y_offset)
        #generate_outline(vertices, faces, [1,1,0,1], 1, 3, round(top_text_baseplate_width), round(top_text_baseplate_height), baseplate_x_scale, baseplate_y_scale, baseplate_x_offset, baseplate_y_offset)

        top_text = config.front_top_text[idx]
        font = "fonts/8bitoperator_jve.ttf"
        font_size = 16
        text_x_scale = desired_size / 55
        text_y_scale = desired_size / 60
        text_x_position = (baseplate_x_offset + 2.5) / text_x_scale
        text_y_position = 0 #(baseplate_x_offset + 2) / text_y_scale
        text_x_size = round((desired_size) / text_x_scale)
        text_y_size = round((desired_size + top_text_baseplate_width) / text_y_scale)
        print(f"({text_x_size}, {text_y_size})")
        generate_text(vertices, faces, top_text, font, font_size, text_x_size, text_y_size, text_x_scale, text_y_scale, text_x_position, text_y_position)

        
        logo_thickness = layer_height * 2
        #generate_logo(vertices, faces, config.current_directory + "logo.png", 8, 8, logo_thickness, [11, 10])

        #'''


        '''
        # For imprinting on silicone mold
        epoxy_edge_length = 4

        qr_code_face_side = False
        
        mold_width = desired_size + (epoxy_edge_length * 2)
        mold_height = desired_size + top_text_baseplate_height + epoxy_edge_length
        mold_x_offset = epoxy_edge_length * 2
        mold_y_offset = epoxy_edge_length * 2

        # Create 4 squares for the loop
        qrcode_mold_with = desired_size + (epoxy_edge_length * 2)
        qrcode_mold_height = desired_size + (epoxy_edge_length * 2) + 1

        outer_side_mold_width = desired_size / 6
        outer_side_mold_height = desired_size / 3

        outer_side_mold_x_offset = qrcode_mold_height + mold_x_offset

        if(qr_code_face_side):
            outer_side_mold_y_offset = mold_y_offset
        else:  
            outer_side_mold_y_offset = mold_y_offset + qrcode_mold_with - outer_side_mold_width

        upper_side_mold_width = qrcode_mold_with - (outer_side_mold_width + top_text_baseplate_height + (epoxy_edge_length * 2))
        upper_side_mold_height = desired_size / 6
        upper_side_x_offset = outer_side_mold_x_offset + outer_side_mold_height - upper_side_mold_height
        
        if(qr_code_face_side):
            upper_side_y_offset = mold_y_offset + outer_side_mold_width
        else:
            upper_side_y_offset = mold_y_offset + mold_width - upper_side_mold_width - outer_side_mold_width

        block_mold_width = top_text_baseplate_height + (epoxy_edge_length * 2)
        block_mold_height = desired_size / 3
        block_mold_x_offset = outer_side_mold_x_offset

        if(qr_code_face_side):
            block_mold_y_offset = upper_side_y_offset + upper_side_mold_width
        else:
            block_mold_y_offset = mold_y_offset

        mold_base_width = round(mold_width + (epoxy_edge_length * 4))
        mold_base_height = round(mold_height + (epoxy_edge_length * 4))
        
        if(qr_code_face_side):
            mold_depth = 20
        else:
            mold_depth = 40

        if(qr_code_face_side):
            mold_wall = 40
        else:
            mold_wall = 60

        print(f"current size: {qrcode_mold_height + outer_side_mold_height}")


        # Base
        generate_base(vertices, faces, layer_height * 5, mold_base_width, mold_base_height, 0, 0)
        
        # Imprent2
        generate_base(vertices, faces, layer_height * mold_depth, outer_side_mold_width, outer_side_mold_height, outer_side_mold_x_offset, outer_side_mold_y_offset)
        generate_base(vertices, faces, layer_height * mold_depth, upper_side_mold_width, upper_side_mold_height, upper_side_x_offset, upper_side_y_offset)
        generate_base(vertices, faces, layer_height * mold_depth, block_mold_width, block_mold_height, block_mold_x_offset, block_mold_y_offset)
        
        # QR Code Mold
        generate_base(vertices, faces, layer_height * mold_depth, qrcode_mold_with, qrcode_mold_height, mold_x_offset, mold_y_offset)


        # Outer wall
        generate_outline(vertices, faces, [1,1,1,1], 3, mold_wall, mold_base_width, mold_base_height, 1, 1, 0, 0)
        #'''

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