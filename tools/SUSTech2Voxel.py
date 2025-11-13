import json
import os
from pathlib import Path

input_folder = "/home/hoangnt/SUSTechPOINTS/data/HAP/label"      
output_folder = "/home/hoangnt/SUSTechPOINTS/data/HAP/PCDet"    

os.makedirs(output_folder, exist_ok=True)

cls_map = {
    "Car": "Car",
    "Person": "Person"
}

for json_file in Path(input_folder).glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        objects = [data]
    elif isinstance(data, list):
        objects = data
    else:
        continue

    lines = []
    for obj in objects:
        psr = obj.get("psr", {})
        pos = psr.get("position", {})
        scale = psr.get("scale", {})
        rot = psr.get("rotation", {})
        cls = cls_map.get(obj.get("obj_type", "Unknown"), "Unknown")

        x, y, z = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)
        dx, dy, dz = scale.get("x", 0), scale.get("y", 0), scale.get("z", 0)
        heading_angle = rot.get("z", 0)

        line = f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {heading_angle:.6f} {cls}"
        lines.append(line)

    out_path = Path(output_folder) / (json_file.stem + ".txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

print("Done! Converted all JSONs to OpenPCDet label format.")
