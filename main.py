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

def insert_color_change(gcode_file, base_thickness, protrusion_thickness):

    # Convert target thickness to string format (e.g., "1.2" or similar)
    target_thickness_str = f";{base_thickness + protrusion_thickness:.1f}"

    # Read the G-code file
    with open(gcode_file, 'r') as file:
        lines = file.readlines()

    # Find all lines matching the target thickness string
    thickness_lines = [i for i, line in enumerate(lines) if target_thickness_str in line]

    # Get the index of the second occurrence of the target thickness
    second_instance_index = thickness_lines[1]

    # Define the color change G-code block
    color_change_gcode = [
        ";COLOR_CHANGE,T0,#50E74C\n",  # The #50E74C is arbitrary and can be any hex color
        "M600 ; Filament color change\n"
    ]

    # Insert the color change G-code block after the second instance of target thickness
    insert_position = second_instance_index + 1
    lines[insert_position:insert_position] = color_change_gcode

    # Output file (overwrite original if output_file not specified)
    with open(gcode_file, 'w') as file:
        file.writelines(lines)

def main():

    # These variables are in milimeters
    desired_size = 45 
    layer_height = 0.2
    protrusion_thickness = layer_height * 2
    base_thickness = layer_height * 5
    base_extension = 13
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
                        #continue
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

        # Scale the text up, then downsize. This prevents loss of resolution.
        text_scale_factor = 2
        font_size = 11
        large_font = ImageFont.truetype("arial.ttf", font_size * text_scale_factor)

        text_width = 500
        text_height = 500
        large_text_image = Image.new('L', (text_width, text_height), color=255)
        
        text_draw = ImageDraw.Draw(large_text_image)

        text_x_position = ((desired_size * idx) + (space_between_qrs * idx) + 2) * text_scale_factor
        text_y_position = (desired_size + 0) * text_scale_factor

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