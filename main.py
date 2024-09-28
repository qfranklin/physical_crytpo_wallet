from PIL import Image, ImageDraw, ImageFont
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import src.create_stl as create_stl
import config.config as config
import src.gcode as gcode

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
    gcode_target_thickness = create_stl.qr_code()
    gcode.generate_gcode()
    gcode.insert_color_change(0.96)
    #create_rear_template(desired_size, 2, base_extension)

if __name__ == "__main__":
    main()