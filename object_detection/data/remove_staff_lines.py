import os
import subprocess
from lxml import etree
from shutil import copyfile

#config
MUSESCORE_CMD = "mscore" 
musicxml_dir = "musicxml_dir"
output_full_dir = "output/full"
output_mask_dir = "output/mask"

os.makedirs(output_full_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

def generate_full_png(xml_path, output_path):
    subprocess.run([
        MUSESCORE_CMD, xml_path,
        "-o", output_path
    ], check=True)

def create_staff_only_version(xml_path, xml_mask_path):
    #Edit MusicXML to only keep staff lines by removing all other elements.
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # remove everything except <staff-lines> and <attributes>
    for part in root.xpath(".//part"):
        for measure in part.xpath(".//measure"):
            for element in measure:
                if element.tag not in ["attributes"]:
                    measure.remove(element)

    tree.write(xml_mask_path)

def process_file(filename):
    name = os.path.splitext(filename)[0]
    xml_path = os.path.join(musicxml_dir, filename)
    full_png = os.path.join(output_full_dir, f"{name}.png")
    mask_xml = os.path.join(musicxml_dir, f"{name}_mask.musicxml")
    mask_png = os.path.join(output_mask_dir, f"{name}_mask.png")

    # save original full PNG
    generate_full_png(xml_path, full_png)

    # create staff-line-only version
    create_staff_only_version(xml_path, mask_xml)
    generate_full_png(mask_xml, mask_png)

    os.remove(mask_xml)

# run over dataset
for file in os.listdir(musicxml_dir):
    if file.endswith(".musicxml") or file.endswith(".xml"):
        print(f"Processing: {file}")
        try:
            process_file(file)
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
