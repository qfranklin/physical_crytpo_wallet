import bpy
import sys
import os
import importlib

# Ensure the script's directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import your modules
import config.config as config
import src.create_stl as create_stl

def main():
    importlib.reload(config)
    importlib.reload(create_stl)

    create_stl.qr_code()

    # For example, clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Path to your STL file
    stl_file = config.current_directory + 'qr.stl'

    # Import the STL file
    bpy.ops.import_mesh.stl(filepath=stl_file)

    #sd_card_bottom_file = config.current_directory + 'sd_card_bottom.stl'

    # Import the STL file
    #bpy.ops.import_mesh.stl(filepath=sd_card_bottom_file)

if __name__ == "__main__":
    main()