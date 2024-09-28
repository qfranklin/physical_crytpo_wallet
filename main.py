from PIL import Image, ImageDraw, ImageFont
import math
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import src.create_stl as create_stl
import config.config as config

def insert_color_change(gcode_file, target_thickness):

    # Convert target thickness to string format (e.g., "1.2" or similar)
    target_thickness_str = f";{target_thickness:.1f}"

    # Read the G-code file
    with open(gcode_file, 'r') as file:
        lines = file.readlines()

    print(f"{target_thickness} {target_thickness_str}")

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
    
def generate_gcode(stl_file, output_gcode, config_file):
    """
    Generates G-code using PrusaSlicer for the provided STL file.
    
    Arguments:
    - stl_file: Path to the input STL file.
    - output_gcode: Path to the output G-code file.
    - config_file: Path to the PrusaSlicer configuration file.
    """

    # Command to run PrusaSlicer via console
    command = [
        prusa_slicer_path, 
        '--export-gcode',
        '--load', config_file,  # Load the configuration file
        '--output', output_gcode,  # Specify the output G-code file
        stl_file  # Specify the STL file to slice
    ]
    
    subprocess.run(command, check=True)

def create_rear_template(square_size_mm, outline_thickness_mm, extension_height_mm):
    # Convert mm to pixels based on DPI
    square_size_px = int((square_size_mm / 25.4) * 300)
    outline_thickness_px = int((outline_thickness_mm / 25.4) * 300)
    extension_height_px = int((extension_height_mm / 25.4) * 300)

    space_between_images = 10
    
    total_qr_codes = len(qr_code_text)
    grid_size = math.ceil(math.sqrt(total_qr_codes))

    img_width = (square_size_px * grid_size) + (space_between_images * (grid_size - 1))
    img_height = ((square_size_px + extension_height_px) * (grid_size - 1)) + (space_between_images * (grid_size - 2))

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", 80)
    bottom_font = ImageFont.truetype("arial.ttf", 40)

    # Loop over the grid and create each QR code image
    for idx, data in enumerate(qr_code_text):
        row = idx // grid_size
        col = idx % grid_size

        # Calculate the top-left position for this image in the grid
        top_left_x = (col * square_size_px) + (col * space_between_images)
        top_left_y = (row * (square_size_px + extension_height_px)) + (row * space_between_images)

        # Create the polygon outline for this QR code
        polygon_points = [
            (top_left_x, top_left_y),
            (top_left_x + square_size_px, top_left_y),
            (top_left_x + square_size_px, top_left_y + square_size_px + extension_height_px),
            (top_left_x + extension_height_px, top_left_y + square_size_px + extension_height_px),
            (top_left_x, top_left_y + square_size_px)
        ]
        draw.polygon(polygon_points, outline="black", fill=None, width=outline_thickness_px)

        # Add the text (front_side_text[idx]) in the top square area
        wrapped_text = []
        current_line = ""

        for char in data:
            test_line = current_line + char
            # Check if adding the next character would exceed the max width
            if draw.textbbox((0, 0), test_line, font=font)[2] <= (square_size_px - (outline_thickness_px * 4)):
                current_line = test_line
            else:
                # If it exceeds, add the current line to wrapped_text and start a new line
                wrapped_text.append(current_line)
                current_line = char  # Start a new line with the current character

        # Add the last line if there is any text left
        if current_line:
            wrapped_text.append(current_line)

        # Calculate total height for the wrapped text
        text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in wrapped_text)

        # Calculate initial y position for centering
        text_y = top_left_y + ((square_size_px - (outline_thickness_px * 2)) - text_height) / 2  # Center text vertically

        # Draw each line of wrapped text
        for line in wrapped_text:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]  # Width of the line
            text_x = top_left_x + (square_size_px - text_width) / 2  # Center text horizontally
            draw.text((text_x, text_y), line, fill="black", font=font)
            text_y += text_bbox[3] - text_bbox[1] + 5  # Move y position for the next line

        bottom_text = config.rear_side_bottom_text[idx]
        bottom_text_bbox = draw.textbbox((0, 0), bottom_text, font=bottom_font)
        bottom_text_width = bottom_text_bbox[2] - bottom_text_bbox[0]

        bottom_text_x = (top_left_x + (square_size_px - bottom_text_width) / 2) + (extension_height_px / 2)
        bottom_text_y = (top_left_y + square_size_px + (extension_height_px - bottom_text_bbox[3]) / 2) - 10

        draw.text((bottom_text_x, bottom_text_y), bottom_text, fill="black", font=bottom_font)

    # Flip the entire image along the vertical axis (if required)
    img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

    # Save the entire image grid to a temporary file
    img.save("temp_image.png", dpi=(300, 300))

    # Create a PDF document with ReportLab
    c = canvas.Canvas("rear_side_outline.pdf", pagesize=letter)
    pdf_width, pdf_height = letter  # Letter size in points (1 point = 1/72 inch)

    # Calculate the size of the entire image in inches
    img_width_inch = img_width / 300  # Convert pixels to inches
    img_height_inch = img_height / 300

    # Convert image size to points (1 inch = 72 points)
    img_width_pt = img_width_inch * 72
    img_height_pt = img_height_inch * 72

    # Center the image on the PDF page
    x_offset = (pdf_width - img_width_pt) / 2
    y_offset = (pdf_height - img_height_pt) / 2

    # Draw the image grid on the PDF
    c.drawImage("temp_image.png", x_offset, y_offset, img_width_pt, img_height_pt)

    # Save the PDF
    c.save()

def main():
    create_stl.qr_code()

    gcode_file = config.current_directory + "qr.gcode"
    prusa_config = config.current_directory + "prusa_slicer_config.ini"
    #generate_gcode(stl_file, gcode_file, prusa_config)
    #insert_color_change(gcode_file, base_thickness + layer_height)
    #create_rear_template(desired_size, 2, base_extension)

if __name__ == "__main__":
    main()