# -- coding: utf-8 --**
import json

import xml.etree.ElementTree as ET

json_path = '../control/raptor.json'

with open(json_path, 'r') as f:
    d = json.loads(f.read())

temple_asset_path = '/assets/Dog/Dog.xml'

temple_xml = ET.parse(temple_asset_path)
root = temple_xml.getroot()

tmp_root = ET.Element('worldbody')
x = tmp_root

# root body

rigid = d['BodyDefs'][0]

body_name = rigid["Name"]
joint_name = body_name + '_y'

l0 = rigid['Param0'] / 2
l1 = rigid['Param2'] / 2  # convert z and y
l2 = rigid['Param1'] / 2
v = rigid['Param0'] * rigid['Param1'] * rigid['Param2']
density = str(int(rigid['Mass'] / v))

x = ET.SubElement(x, 'body', name=body_name, pos='0 0 0')
ET.SubElement(x, 'freejoint', name=joint_name)
ET.SubElement(x, 'geom', type='box', size=f'{l0} {l1} {l2}', density=density)

body_list = [x]

# body trees
for i in range(1, len(d['Skeleton']['Joints'])):
    joint = d['Skeleton']['Joints'][i]
    rigid = d['BodyDefs'][i]

    parent_id = joint['Parent']

    body_name = rigid["Name"]
    joint_name = body_name + '_y'

    l0 = rigid['Param0'] / 2
    l1 = rigid['Param2'] / 2  # convert z and y
    l2 = rigid['Param1'] / 2
    v = rigid['Param0'] * rigid['Param1'] * rigid['Param2']
    density = str(int(rigid['Mass'] / v))

    x = body_list[parent_id]
    x = ET.SubElement(x, 'body', name=body_name, pos=f'{joint["AttachX"]} {joint["AttachZ"]} {joint["AttachY"]}')
    ET.SubElement(x, 'joint', name=joint_name, axis='0 1 0', range=f'{joint["LimLow"]} {joint["LimHigh"]}')
    ET.SubElement(x, 'geom', type='box', pos=f'{rigid["AttachX"]} {rigid["AttachZ"]} {rigid["AttachY"]}',
                  size=f'{l0} {l1} {l2}', density=density)

    body_list.append(x)

root[6][3] = tmp_root[0]
temple_xml.write('output.xml')
