import qrcode
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math
import sys
import os

if '__file__' in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Provide a default directory when running within Blender
    script_dir = os.path.dirname(bpy.data.filepath)



# Add the directory of config.py to sys.path to make sure Blender can detect it
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import config

try:
    import bpy  
    import importlib
    is_blender_env = True
    importlib.reload(config) # Cache bust Blender
except ImportError:
    is_blender_env = False


cls = lambda: os.system('cls'); 
cls()

def main():

    # These are in milimeters
    desired_size = 46 
    qr_thickness = 0.28
    base_thickness = 1.12
    base_extension = 15
    space_between_qrs = 5

    all_vertices = []
    all_faces = []

    # Calculate grid layout dynamically
    total_qr_codes = len(config.grid_data)
    grid_size = math.ceil(math.sqrt(total_qr_codes))

    # Loop over the list of public/private keys
    for idx, data in enumerate(config.grid_data):
        # Determine row and column position for each QR code
        col = idx // grid_size
        row = idx % grid_size
        x_offset = col * (desired_size + base_extension + (space_between_qrs * idx))
        y_offset = row * (desired_size + (space_between_qrs * idx))

        # Generate QR Code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=5, # 4 units is the qr code standard, but add one to account for outline
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill='black', back_color='white')
        img = img.convert('L')  # Convert to grayscale
        pixels = np.array(img)  # Get pixel data as a numpy array
        
        height, width = pixels.shape

        # Calculate scaling factors
        x_scale = desired_size / width
        y_scale = desired_size / height
        z_scale = qr_thickness  # Extrusion height

        # Prepare vertices and faces for STL
        vertices = []
        faces = []

        # Add baseplate vertices and faces
        vertices.extend([
            [x_offset, y_offset, 0],
            [x_offset + width * x_scale, y_offset, 0],
            [x_offset + width * x_scale, y_offset + height * y_scale, 0],
            [x_offset, y_offset + height * y_scale, 0],
            [x_offset, y_offset, base_thickness],
            [x_offset + width * x_scale, y_offset, base_thickness],
            [x_offset + width * x_scale, y_offset + height * y_scale, base_thickness],
            [x_offset, y_offset + height * y_scale, base_thickness]
        ])
        faces.extend([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Side faces
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7]
        ])

        # Define QR code vertices and faces for each pixel
        for y in range(height):
            for x in range(width):
                if pixels[y, x] < 128:  # Black pixels only (for QR code)
                    z = z_scale  # Set height for black pixels
                elif x == 0 or y == 0 or (x + 1) == width or (y + 1) == height:
                    z = z_scale / 2
                else:
                    z = 0  # Set flat for white pixels

                qr_idx = len(vertices)

                vertices.extend([
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, base_thickness],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness],
                    [x * x_scale + x_offset, y * y_scale + y_offset, base_thickness + z],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, base_thickness + z],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + z],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, base_thickness + z]
                ])

                # Create faces for the cube (6 faces per cube)
                faces.extend([
                    [qr_idx, qr_idx + 1, qr_idx + 5], [qr_idx, qr_idx + 5, qr_idx + 4],
                    [qr_idx + 1, qr_idx + 2, qr_idx + 6], [qr_idx + 1, qr_idx + 6, qr_idx + 5],
                    [qr_idx + 2, qr_idx + 3, qr_idx + 7], [qr_idx + 2, qr_idx + 7, qr_idx + 6],
                    [qr_idx + 3, qr_idx, qr_idx + 4], [qr_idx + 3, qr_idx + 4, qr_idx + 7],
                    [qr_idx + 4, qr_idx + 5, qr_idx + 6], [qr_idx + 4, qr_idx + 6, qr_idx + 7]
                ])

        base_extension_height = int(round(desired_size / x_scale, 0))
        base_extension_width = int(round(base_extension / y_scale, 0))
        adjacency_range = base_extension_width

        # Add the base extension
        for y in range(base_extension_height):
            for x in range(base_extension_width):

                if y == 0 or \
                  (x + 1) == base_extension_width or \
                  (y + 1) == base_extension_height or \
                  (((base_extension_width - 1 - x) + (base_extension_height - 1 - y)) == adjacency_range - 1):
                    z = z_scale / 2
                else:
                    z = 0

                if ((base_extension_width - 1 - x) + (base_extension_height - 1 - y)) < adjacency_range - 1:
                    continue

                qr_idx = len(vertices)

                if ((base_extension_width - 1 - x) + (base_extension_height - 1 - y)) == adjacency_range:
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
                elif ((base_extension_width - 1 - x) + (base_extension_height - 1 - y)) == adjacency_range - 1:

                    if (y + 1) == base_extension_height:
                        vertices.extend([
                            [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                    elif (x + 1) == base_extension_width: 
                        vertices.extend([
                            [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, (y - 1) * y_scale + y_offset, base_thickness + z],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                    else:
                        vertices.extend([
                            [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                            [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                            [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                            [(x - 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                        ])
                else:
                    vertices.extend([
                        [x * x_scale + desired_size, y * y_scale + y_offset, 0],
                        [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, 0],
                        [(x + 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, 0],
                        [x * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + 1) * x_scale + desired_size, y * y_scale + y_offset, base_thickness + z],
                        [(x + 1) * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z],
                        [x * x_scale + desired_size, (y + 1) * y_scale + y_offset, base_thickness + z]
                    ])
                    
                # Create faces for the cube (6 faces per cube)
                faces.extend([
                    [qr_idx, qr_idx + 1, qr_idx + 2], [qr_idx, qr_idx + 2, qr_idx + 3],  # Bottom face
                    [qr_idx + 4, qr_idx + 5, qr_idx + 6], [qr_idx + 4, qr_idx + 6, qr_idx + 7],  # Top face
                    [qr_idx, qr_idx + 1, qr_idx + 5], [qr_idx, qr_idx + 5, qr_idx + 4],  # Front face
                    [qr_idx + 2, qr_idx + 3, qr_idx + 7], [qr_idx + 2, qr_idx + 7, qr_idx + 6],  # Back face
                    [qr_idx + 1, qr_idx + 2, qr_idx + 6], [qr_idx + 1, qr_idx + 6, qr_idx + 5],  # Right face
                    [qr_idx + 3, qr_idx + 0, qr_idx + 4], [qr_idx + 3, qr_idx + 4, qr_idx + 7]   # Left face
                ])

        # Next section is for adding text to the bottom of the qr code. 

        # Scale the text up, then downsize. This prevents loss of resolution.
        text_scale_factor = 2
        font_size = 11
        large_font = ImageFont.truetype("arial.ttf", font_size * text_scale_factor)

        text_width = 500
        text_height = 500
        large_text_image = Image.new('L', (text_width, text_height), color=255)
        
        text_draw = ImageDraw.Draw(large_text_image)

        text_x_position = ((desired_size * idx) + (space_between_qrs * idx) + 2) * text_scale_factor
        text_y_position = (desired_size + 1) * text_scale_factor

        # Draw the text on the new larger image
        text_draw.text((text_x_position, text_y_position), config.text[idx], fill=0, font=large_font)

        # Step 3: Resize the image down to the desired final size
        # The text image is created larger, so now we reduce the size for better resolution.
        text_image = large_text_image.resize(
            (text_width // text_scale_factor, text_height // text_scale_factor),
            Image.Resampling.LANCZOS
        )

        # Optional image transformations: mirror and rotate (depends on your use case)
        text_image = ImageOps.mirror(text_image)
        text_image = text_image.rotate(90, expand=True)

        # Convert this text image to 3D vertices (black pixels = protruding)
        text_pixels = text_image.load()

        # Loop through the pixels in the text image and generate vertices
        for y in range(text_image.height):
            for x in range(text_image.width):
                if text_pixels[x, y] < 128:  # Black pixels = protruding areas
                    text_idx = len(vertices)

                    # Create vertices without the QR code offset, apply your own scaling instead
                    vertices.extend([
                        [x, y, base_thickness + qr_thickness],  # Top-left
                        [(x + 1), y, base_thickness + qr_thickness],  # Top-right
                        [(x + 1), (y + 1), base_thickness + qr_thickness],  # Bottom-right
                        [x, (y + 1), base_thickness + qr_thickness],  # Bottom-left
                        [x, y, base_thickness],  # Base top-left
                        [(x + 1), y, base_thickness],  # Base top-right
                        [(x + 1), (y + 1), base_thickness],  # Base bottom-right
                        [x, (y + 1), base_thickness]  # Base bottom-left
                    ])

                    # Define faces for each pixel (each square face consists of two triangles)
                    faces.extend([
                        [text_idx, text_idx + 1, text_idx + 2],
                        [text_idx, text_idx + 2, text_idx + 3],
                        [text_idx, text_idx + 1, text_idx + 5],
                        [text_idx + 5, text_idx + 4, text_idx],
                        [text_idx + 1, text_idx + 2, text_idx + 6],
                        [text_idx + 1, text_idx + 6, text_idx + 5],
                        [text_idx + 2, text_idx + 3, text_idx + 7],
                        [text_idx + 2, text_idx + 7, text_idx + 6],
                        [text_idx + 3, text_idx, text_idx + 4],
                        [text_idx + 3, text_idx + 4, text_idx + 7]
                    ])

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

    # Save as STL
    qr_mesh.save(rf'{config.current_directory}qr_code.stl')

    if is_blender_env:

        # Path to your STL file
        stl_file_path = config.current_directory + "qr_code.stl"

        # Clear existing objects in the scene
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

        # Import the STL file
        bpy.ops.import_mesh.stl(filepath=stl_file_path)

        # Add a camera if there isn't one
        if 'Camera' not in bpy.data.objects:
            bpy.ops.object.camera_add(location=(0, -5, 5))
        camera = bpy.data.objects['Camera']
        camera.rotation_euler = (1.1, 0, 0)  # Adjust the rotation as needed
        bpy.context.scene.camera = camera

if __name__ == "__main__":
    main()