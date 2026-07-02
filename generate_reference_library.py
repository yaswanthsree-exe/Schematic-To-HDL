import os
import schemdraw
import schemdraw.logic as logic

REF_DIR = "reference_library"
os.makedirs(REF_DIR, exist_ok=True)

gates = {
    "AND": logic.And(),
    "OR": logic.Or(),
    "XOR": logic.Xor(),
    "NAND": logic.Nand(),
    "NOR": logic.Nor(),
    "NOT": logic.Not()
}

def generate_reference_library():
    print("Generating Tightly Cropped High-Quality Logic Gate PNGs...")
    
    for name, gate in gates.items():
        with schemdraw.Drawing(show=False) as d:
            d.config(fontsize=14, lw=2.5)
            d.add(gate.right().at((0, 0)))
            
            img_path = os.path.join(REF_DIR, f"{name}.png")
            # bbox_inches='tight' crops it perfectly to the ink
            d.save(img_path, transparent=True, dpi=150)
            
            print(f"Saved {name} to {img_path}")

if __name__ == "__main__":
    generate_reference_library()
