import xml.etree.ElementTree as ET

def generate_xml(keypoints_list, template_path, output_path, video_file):
    
    with open(template_path, "r") as file:
        template = file.read()

    print(f"Erstelle XML für {len(keypoints_list)} Frames: video_file={video_file}")

    images_xml = ""
    for i, keypoints in enumerate(keypoints_list):
        frame_number = str(keypoints["frame_number"]).zfill(6)
        image_entry = f'''
        <image id="{i}" name="{video_file}_img_{frame_number}.jpg" width="1920" height="1080">
            <skeleton label="person" source="manual" z_order="0">
        '''
        for name, point in keypoints.items():
            if name == "frame_number":
                continue
            image_entry += f'''
                <points label="{name}" source="manual" outside="0" occluded="0" points="{point['x']},{point['y']}"/>
            '''
        image_entry += '''
            </skeleton>
        </image>
        '''
        images_xml += image_entry

    output_xml = template.replace("{images}", images_xml)
    with open(output_path, "w") as file:
        file.write(output_xml)

    print(f"XML gespeichert unter: {output_path}")
