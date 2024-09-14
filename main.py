import qrcode
import numpy as np
from stl import mesh
from PIL import Image
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

try:
    import bpy  # Blender's Python module
    is_blender_env = True
except ImportError:
    is_blender_env = False

import config


print(f"Blender: {is_blender_env}")


from os import system; 
cls = lambda: system('cls'); 
cls()

def generate_qr_code_mesh(data, x_offset, y_offset):
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

    # Define baseplate vertices with offset included
    base_vertices = [
        [x_offset, y_offset, 0],
        [x_offset + width * x_scale, y_offset, 0],
        [x_offset + width * x_scale, y_offset + height * y_scale, 0],
        [x_offset, y_offset + height * y_scale, 0],
        [x_offset, y_offset, config.base_thickness],
        [x_offset + width * x_scale, y_offset, config.base_thickness],
        [x_offset + width * x_scale, y_offset + height * y_scale, config.base_thickness],
        [x_offset, y_offset + height * y_scale, config.base_thickness]
    ]

    idx = len(vertices)
    
    # Add baseplate vertices and faces
    vertices.extend(base_vertices)
    faces.extend([
        [0, idx + 1, idx + 2], [0, idx + 2, idx + 3], 
        [idx + 4, idx + 5, idx + 6], [idx + 4, idx + 6, idx + 7],
        [0, idx + 1, idx + 5], [0, idx + 5, idx + 4],
        [idx + 1, idx + 2, idx + 6], [idx + 1, idx + 6, idx + 5],
        [idx + 2, idx + 3, idx + 7], [idx + 2, idx + 7, idx + 6],
        [idx + 3, 0, idx + 4], [idx + 3, idx + 4, idx + 7]
    ])

    # Define QR code vertices and faces for each pixel
    for y in range(height):
        for x in range(width):
            if pixels[y, x] < 128:  # Black pixels only (for QR code)
                z = z_scale  # Set height for black pixels
            else:
                z = 0  # Set flat for white pixels

            idx = len(vertices)

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
                [idx, idx + 1, idx + 5], [idx, idx + 5, idx + 4],
                [idx + 1, idx + 2, idx + 6], [idx + 1, idx + 6, idx + 5],
                [idx + 2, idx + 3, idx + 7], [idx + 2, idx + 7, idx + 6],
                [idx + 3, idx, idx + 4], [idx + 3, idx + 4, idx + 7],
                [idx + 4, idx + 5, idx + 6], [idx + 4, idx + 6, idx + 7]
            ])

    return vertices, faces

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

        # Generate QR code mesh for the current data
        vertices, faces = generate_qr_code_mesh(data, x_offset, y_offset)

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

    print("executed 1")

    if is_blender_env:


        print("executed")

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

print(__name__ == "__main__")

if __name__ == "__main__":
    main()