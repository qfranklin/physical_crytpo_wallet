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


from os import system; 
cls = lambda: system('cls'); 
cls()

def main():
    # Initialize overall vertices and faces for all QR codes
    all_vertices = []
    all_faces = []

    # Calculate grid layout dynamically
    total_qr_codes = len(config.grid_data)
    grid_size = math.ceil(math.sqrt(total_qr_codes))  # Calculate grid size (rows and columns)

    # Loop over the list of strings (grid_data)
    for idx, data in enumerate(config.grid_data):
        # Determine row and column position for each QR code
        row = idx // grid_size
        col = idx % grid_size
        x_offset = col * (config.desired_size + config.space_between_qrs)
        y_offset = row * (config.desired_size + config.space_between_qrs)

        x_offset += (config.base_extension * col)

        # Generate QR Code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=config.box_size,
            border=4,  # Border size in boxes
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Save the QR code as an image (temp image in memory)
        img = qr.make_image(fill='black', back_color='white')
        img = img.convert('L')  # Convert to grayscale
        pixels = np.array(img)  # Get pixel data as a numpy array
        
        height, width = pixels.shape

        # Calculate scaling factors
        x_scale = config.desired_size / width
        y_scale = config.desired_size / height
        z_scale = config.qr_thickness  # Extrusion height

        # Prepare vertices and faces for STL
        vertices = []
        faces = []

        #'''
        # Add baseplate vertices and faces
        vertices.extend([
            [x_offset, y_offset, 0],
            [x_offset + width * x_scale + config.base_extension, y_offset, 0],
            [x_offset + width * x_scale + config.base_extension, y_offset + height * y_scale, 0],
            [x_offset, y_offset + height * y_scale, 0],
            [x_offset, y_offset, config.base_thickness],
            [x_offset + width * x_scale + config.base_extension, y_offset, config.base_thickness],
            [x_offset + width * x_scale + config.base_extension, y_offset + height * y_scale, config.base_thickness],
            [x_offset, y_offset + height * y_scale, config.base_thickness]
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
                else:
                    z = 0  # Set flat for white pixels

                qr_idx = len(vertices)

                vertices.extend([
                    [x * x_scale + x_offset, y * y_scale + y_offset, config.base_thickness],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, config.base_thickness],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, config.base_thickness],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, config.base_thickness],
                    [x * x_scale + x_offset, y * y_scale + y_offset, config.base_thickness + z],
                    [(x + 1) * x_scale + x_offset, y * y_scale + y_offset, config.base_thickness + z],
                    [(x + 1) * x_scale + x_offset, (y + 1) * y_scale + y_offset, config.base_thickness + z],
                    [x * x_scale + x_offset, (y + 1) * y_scale + y_offset, config.base_thickness + z]
                ])

                # Create faces for the cube (6 faces per cube)
                faces.extend([
                    [qr_idx, qr_idx + 1, qr_idx + 5], [qr_idx, qr_idx + 5, qr_idx + 4],
                    [qr_idx + 1, qr_idx + 2, qr_idx + 6], [qr_idx + 1, qr_idx + 6, qr_idx + 5],
                    [qr_idx + 2, qr_idx + 3, qr_idx + 7], [qr_idx + 2, qr_idx + 7, qr_idx + 6],
                    [qr_idx + 3, qr_idx, qr_idx + 4], [qr_idx + 3, qr_idx + 4, qr_idx + 7],
                    [qr_idx + 4, qr_idx + 5, qr_idx + 6], [qr_idx + 4, qr_idx + 6, qr_idx + 7]
                ])
        #'''


        # Example usage:
        font_path = config.current_directory + "text_font.ttf"  # Provide path to your font file

        print(f"{height} {width}")

        #'''

        config_width = 200
        config_height = 50

        # Create an image for text
        text_image = Image.new('L', (config_width, config_height), color=255)  # White background
        text_draw = ImageDraw.Draw(text_image)
        
        # Load a font
        font = ImageFont.truetype(font_path, config.font_size)
        
        # Get text bounding box (replaces textsize)
        text_bbox = text_draw.textbbox((0, 0), config.text[idx], font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        # Position the text in the center
        text_position = ((config_width - text_width) // 2, (config_height - text_height) // 2)
        
        # Draw the text onto the image
        text_draw.text(text_position, config.text[idx], fill=0, font=font)  # Black text

        text_image = ImageOps.mirror(text_image)
        text_image = text_image.rotate(90, expand=True)

        # Now convert this text image to 3D vertices (black pixels = protruding)
        text_pixels = text_image.load()
        text_z_height = 1.5  # Height of the protruding text
        text_x_offset = x_offset  # Align with the baseplate

        for y in range(text_image.height):
            for x in range(text_image.width):
                if text_pixels[x, y] < 128:  # Black pixels
                    z = config.base_thickness + text_z_height

                    # Define vertices for the current pixel
                    text_idx = len(vertices)                    
                    
                    vertices.extend([
                        [text_x_offset + x * x_scale, y_offset + y * y_scale, z],
                        [text_x_offset + (x + 1) * x_scale, y_offset + y * y_scale, z],
                        [text_x_offset + (x + 1) * x_scale, y_offset + (y + 1) * y_scale, z],
                        [text_x_offset + x * x_scale, y_offset + (y + 1) * y_scale, z],
                        [text_x_offset + x * x_scale, y_offset + y * y_scale, config.base_thickness],
                        [text_x_offset + (x + 1) * x_scale, y_offset + y * y_scale, config.base_thickness],
                        [text_x_offset + (x + 1) * x_scale, y_offset + (y + 1) * y_scale, config.base_thickness],
                        [text_x_offset + x * x_scale, y_offset + (y + 1) * y_scale, config.base_thickness]
                    ])
                    
                    # Define faces for the current pixel
                    faces.extend([
                        # Top face
                        [text_idx, text_idx + 1, text_idx + 2],
                        [text_idx, text_idx + 2, text_idx + 3],
                        # Side faces
                        [text_idx, text_idx + 1, text_idx + 5],
                        [text_idx, text_idx + 5, text_idx + 4],
                        [text_idx + 1, text_idx + 2, text_idx + 6],
                        [text_idx + 1, text_idx + 6, text_idx + 5],
                        [text_idx + 2, text_idx + 3, text_idx + 7],
                        [text_idx + 2, text_idx + 7, text_idx + 6],
                        [text_idx + 3, text_idx, text_idx + 4],
                        [text_idx + 3, text_idx + 4, text_idx + 7]
                    ])

        # Append to overall vertices and faces list
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