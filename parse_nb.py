import json

with open("Final_Pollution_Project.ipynb", "r") as f:
    nb = json.load(f)
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            print(f"# Cell {i}")
            print("".join(cell.get('source', [])))
            print("\n" + "-"*40 + "\n")
