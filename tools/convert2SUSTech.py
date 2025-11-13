import os
import json
from pathlib import Path

def convert_txt_to_json_multi(input_folder, output_folder):
    label_map = {
        1: "Car",
        2: "Truck", 
        3: "Construction_Vehicle",
        4: "Bus",
        5: "Trailer",
        6: "Barrier",
        7: "Motorcycle",
        8: "Bicycle",
        9: "Pedestrian",
        10: "Traffic_Cone"
    }

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_folder.glob("*.txt"))
    if not txt_files:
        return

    for txt_file in txt_files:
        output_json = output_folder / (txt_file.stem + ".json")
        output = []

        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue

                # File format: x y z l w h yaw class
                x, y, z, l, w, h, yaw, label = map(float, parts[:8])
                label = int(label)
                
                obj = {
                    "obj_id": str(i),
                    "obj_type": label_map.get(label, f"Unknown_{label}"),
                    "psr": {
                        "position": {"x": float(x), "y": float(y), "z": float(z)},
                        "rotation": {"x": 0.0, "y": 0.0, "z": float(yaw)},
                        "scale": {
                            "x": float(l),  
                            "y": float(w),  
                            "z": float(h)   
                        }
                    }
                }
                output.append(obj)
        
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"{txt_file.name} → {output_json.name}: {len(output)} objects converted")


convert_txt_to_json_multi("res", "res_json")
