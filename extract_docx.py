import docx
import sys

def extract():
    doc = docx.Document('Quantum_LPR_Research_Paper.docx')
    with open('doc_content.txt', 'w', encoding='utf-8') as f:
        for p in doc.paragraphs:
            f.write(p.text + '\n')

if __name__ == "__main__":
    extract()
