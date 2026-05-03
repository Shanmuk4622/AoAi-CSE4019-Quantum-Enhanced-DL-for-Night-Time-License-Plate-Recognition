import docx
doc = docx.Document('Quantum_LPR_Research_Paper.docx')
for i, p in enumerate(doc.paragraphs):
    if 'Figure' in p.text or 'Fig.' in p.text:
        print(f'Paragraph {i}: {p.text}')
    for r in p.runs:
        if 'graphic' in r._element.xml:
            print(f'Paragraph {i} contains an image')
