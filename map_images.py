import docx
doc = docx.Document('Quantum_LPR_Research_Paper.docx')
for i, p in enumerate(doc.paragraphs):
    for r in p.runs:
        if 'graphic' in r._element.xml:
            # find the r:embed attribute
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r._element.xml)
            for blip in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                embed_id = blip.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed']
                target = doc.part.rels[embed_id].target_ref
                print(f'Paragraph {i} has image: {target}')
