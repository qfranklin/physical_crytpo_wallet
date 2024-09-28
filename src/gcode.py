import subprocess
import config.config as config

def generate_gcode():
    """
    Generates G-code using PrusaSlicer for the provided STL file.
    
    Arguments:
    - stl_file: Path to the input STL file.
    - output_gcode: Path to the output G-code file.
    - config_file: Path to the PrusaSlicer configuration file.
    """

    # Command to run PrusaSlicer via console
    command = [
        config.prusa_slicer_path, 
        '--export-gcode',
        '--load', "prusa_slicer_config.ini",  # Load the configuration file
        '--output', "qr.gcode",  # Specify the output G-code file
        config.current_directory + 'qr.stl'  # Specify the STL file to slice
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def insert_color_change(target_thickness):

    # Convert target thickness to string format (e.g., "1.2" or similar)
    target_thickness_str = f";{target_thickness:.2f}"

    # Read the G-code file
    with open(config.current_directory + 'qr.gcode', 'r') as file:
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
    with open(config.current_directory + 'qr.gcode', 'w') as file:
        file.writelines(lines)
    