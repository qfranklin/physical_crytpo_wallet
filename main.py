import src.create_stl as create_stl
import src.gcode as gcode
import src.rear_side_template as rear_side_template

def main():
    gcode_target_thickness = create_stl.qr_code()
    gcode.generate_gcode()
    gcode.insert_color_change(gcode_target_thickness)
    rear_side_template.create(45, 2, 14)

if __name__ == "__main__":
    main()