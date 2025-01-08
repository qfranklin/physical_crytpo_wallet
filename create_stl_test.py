top_text = config.front_top_text[idx]
font = "fonts/8bitoperator_jve.ttf"
font_size = 16

# Calculate text scales based on desired size
text_x_scale = desired_size / 100
text_y_scale = desired_size / 100

text_x_position = (baseplate_x_offset + 2.5) / text_x_scale
text_y_position = 0
text_x_size = round((desired_size) / text_x_scale)
text_y_size = round((desired_size + top_text_baseplate_width) / text_y_scale)

print(f"({text_x_size}, {text_y_size})")
generate_text(vertices, faces, top_text, font, font_size, text_x_size, text_y_size, text_x_scale, text_y_scale, text_x_position, text_y_position)