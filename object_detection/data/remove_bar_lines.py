import xml.etree.ElementTree as ET
import subprocess
import os

from pathlib import Path
from xml.dom import minidom

APPEARANCE_BLOCK = '''
<appearance>
  <line-width type="light barline">1.8</line-width>
  <line-width type="heavy barline">5.5</line-width>
  <line-width type="beam">5</line-width>
  <line-width type="bracket">4.5</line-width>
  <line-width type="dashes">1</line-width>
  <line-width type="enclosure">1</line-width>
  <line-width type="ending">1.1</line-width>
  <line-width type="extend">1</line-width>
  <line-width type="leger">1.6</line-width>
  <line-width type="pedal">1.1</line-width>
  <line-width type="octave shift">1.1</line-width>
  <line-width type="slur middle">2.1</line-width>
  <line-width type="slur tip">0.5</line-width>
  <line-width type="staff">0</line-width>
  <line-width type="stem">1</line-width>
  <line-width type="tie middle">2.1</line-width>
  <line-width type="tie tip">0.5</line-width>
  <line-width type="tuplet bracket">1</line-width>
  <line-width type="wedge">1.2</line-width>
  <note-size type="cue">70</note-size>
  <note-size type="grace">70</note-size>
  <note-size type="grace-cue">49</note-size>
</appearance>
'''

import zipfile
from xml.etree import ElementTree as ET
from pathlib import Path

def extract_and_save_musicxml_from_mxl(mxl_path, output_path):
    try:
        with zipfile.ZipFile(mxl_path, 'r') as archive:
            xml_candidates = [name for name in archive.namelist() if name.endswith(".xml") or name.endswith(".musicxml")]

            for filename in xml_candidates:
                with archive.open(filename) as file:
                    xml_bytes = file.read()
                    try:
                        root = ET.fromstring(xml_bytes)
                        if root.tag in {"score-partwise", "score-timewise"}:
                            # Save it to disk
                            with open(output_path, "wb") as out_file:
                                out_file.write(xml_bytes)
                            return True
                    except ET.ParseError:
                        continue  # skip non-parseable files

    except zipfile.BadZipFile:
        print(f"Not a valid zip archive: {mxl_path}")
    except Exception as e:
        print(f"Unexpected error reading {mxl_path}: {e}")
    
    return False  # No valid MusicXML file found


def process_xml(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Remove all <barline> tags with a location attribute
    for part in root.findall('part'):
        for measure in part.findall('measure'):
            for barline in list(measure.findall('barline')):
                if 'location' in barline.attrib:
                    measure.remove(barline)

    # Insert or update <appearance> inside <defaults>
    defaults = root.find('defaults')
    appearance_elem = ET.fromstring(APPEARANCE_BLOCK)

    if defaults is not None:
        for old in defaults.findall('appearance'):
            defaults.remove(old)
        defaults.append(appearance_elem)
    else:
        defaults = ET.Element('defaults')
        defaults.append(appearance_elem)
        root.insert(0, defaults)

    # Modify <staff-details> in the first measure with <attributes>
    for part in root.findall('part'):
        for measure in part.findall("measure"):
            attr = measure.find("attributes")
            if attr is not None:
                # Remove existing staff-details
                for sd in list(attr.findall("staff-details")):
                    attr.remove(sd)
                for staff_num in range(1, 3):
                    staff_details = ET.SubElement(attr, "staff-details")
                    staff_details.set("number", str(staff_num))
                    
                    sl = ET.SubElement(staff_details, "staff-lines")
                    sl.text = "5"

                    # Disable each line
                    for i in range(1, 6):
                        ld = ET.SubElement(staff_details, "line-detail")
                        ld.set("line", str(i))
                        ld.set("print-object", "no")

                break  # Only need to do this for the first <attributes> element
        break  # Only apply to the first part

    # Pretty-print the updated XML
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Write to file, cleaning blank lines
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join([line for line in pretty_xml.split('\n') if line.strip()]))

def convert_musicxml_to_pdf(musescore_path, input_file, output_pdf):
    try:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen" 

        subprocess.run([
            musescore_path,
            input_file,
            "-o",
            output_pdf
        ], env=env, check=True)
        print(f"PDF exported to: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting PDF: {e}")

if __name__ == "__main__":
    musescore_path = "/home/chomba/downloads/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
    # process_xml("test.xml", "test_output.musicxml")
    # convert_musicxml_to_pdf(musescore_path, "test_output.musicxml", "out.png")

    mxl_dir = Path("output")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print("dog")

    for file in mxl_dir.glob("*.mxl"):
        output_xml = output_dir / f"{file.stem}.musicxml"
        success = extract_and_save_musicxml_from_mxl(file, output_xml)

        if success:
            process_xml(output_xml, output_xml)
            convert_musicxml_to_pdf(musescore_path, str(output_xml), str(output_xml.with_suffix(".png")))
        else:
            print(f" Skipped: {file.name}")
