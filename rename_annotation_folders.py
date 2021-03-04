from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import glob
import os

parser = ArgumentParser()
parser.add_argument("--infolder", type=Path)

if __name__ == "__main__":
    args = parser.parse_args()
    for fold in tqdm(glob.glob(os.path.join(args.infolder, '*'))):
        print(f'Changing folder {fold}')
        for fn in Path(fold).iterdir():
            if fn.is_dir():
                new_name = fn.name.replace(" - ", "_")
                new_name = new_name.replace(" ", "")
                prop_file = fn/"Slidedat.ini"
                try:
                    with prop_file.open("r") as f:
                        lines = f.read().split("\n")
                except FileNotFoundError:
                    print('Not a slide folder')
                    break
                lines.insert(3, "OBJECTIVE_MAGNIFICATION = 20")
                line = lines.pop(2)
                line = line.replace(" - ", "_")
                try:
                    number = line.split(" ")[3]
                    line = line.replace(f" {number}", f"{number}")
                    lines.insert(2, line)
                except IndexError:
                    break
                with prop_file.open("w") as f:
                    f.write("\n".join(lines))
            else:
                new_name = fn.name.replace(" - ", "_")
                new_name = new_name.replace(" ", "")
            fn.rename(fn.parent/new_name)