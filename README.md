# QR Code to 3D Mesh Generator

This script generates 3D meshes from QR codes and saves them as STL files. It can be used within Blender to create and visualize these meshes.

## Requirements

1. **Python Dependencies**:
   - `qrcode`
   - `numpy`
   - `Pillow`
   - `numpy-stl`

   You can install these dependencies using `pip`:

   ```bash
   pip install qrcode numpy Pillow numpy-stl

2. **Setup**:
   ```bash
   git clone <repository-url>
   cd qr_code_python
   cp config.py.example config.py

2. **Usage**:
   Terminal
   ```bash
   py main.py
   ```

   Blender Terminal
   ```bash
   exec(open("path\\to\\qr_code_python\\main.py").read(), {'__file__': "path\\to\\qr_code_python\\main.py", '__name__': '__main__'})
   ```

## Citation
Text font obtained from https://www.fontspace.com/8-bit-font-f7996