import re

def main():
    with open('ACCESS_latex_template_20240429/access.tex', 'r', encoding='utf-8') as f:
        access_tex = f.read()

    # Extract elements
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', access_tex, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    keywords_match = re.search(r'\\begin\{keywords\}(.*?)\\end\{keywords\}', access_tex, re.DOTALL)
    keywords = keywords_match.group(1).strip().replace(',', ' \\sep ') if keywords_match else ""
    
    body_start = access_tex.find('\\section{Introduction}')
    body_end = access_tex.find('\\begin{thebibliography}')
    body = access_tex[body_start:body_end].strip() if body_start != -1 else ""

    def repl_figure(m):
        macro = m.group(1) # Figure or Figure*
        filename = m.group(2)
        caption = m.group(3)
        fig_env = "figure*" if "Figure*" in macro else "figure"
        width = "\\textwidth" if "Figure*" in macro else "\\columnwidth"
        
        return f"""\\begin{{{fig_env}}}[htbp]
\\centering
\\includegraphics[width={width}]{{{filename}}}
\\caption{{{caption}}}
\\end{{{fig_env}}}"""

    body = re.sub(r'\\(Figure\*?)(?:\[.*?\])?(?:\(.*?\)?)?(?:\[.*?\])?\{(.*?)\}\s*\{(.*?)\}', repl_figure, body, flags=re.DOTALL)

    bib_start = access_tex.find('\\begin{thebibliography}')
    bib_end = access_tex.find('\\end{thebibliography}') + len('\\end{thebibliography}')
    bibliography = access_tex[bib_start:bib_end] if bib_start != -1 else ""

    # Build the entire Elsarticle document
    final_tex = f"""\\documentclass[preprint,12pt]{{elsarticle}}
\\usepackage{{amssymb}}
\\usepackage{{amsmath}}
\\usepackage{{tabularx}}
\\usepackage{{booktabs}}
\\usepackage{{bm}}

\\journal{{Pattern Recognition}}

\\begin{{document}}

\\begin{{frontmatter}}

\\title{{Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition: A Hybrid Quantum-Classical Neural Network with 8-Qubit Variational Circuit, Zero-DCE Enhancement, and CTC Sequence Decoding}}

\\author[1]{{Sreenivasa Reddy Edara\\corref{{cor1}}}}
\\ead{{sreenivasareddy.e@vitap.ac.in}}

\\author[1]{{Shanmukesh Bonala}}
\\ead{{Shanmukesh.23BCE20070@vitapstudent.ac.in}}

\\cortext[cor1]{{Corresponding author}}

\\affiliation[1]{{organization={{School of Computer Science and Engineering (SCOPE), VIT-AP University}},
            addressline={{Amaravati}},
            city={{Amaravati}},
            postcode={{522241}},
            state={{Andhra Pradesh}},
            country={{India}}}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{keyword}}
{keywords}
\\end{{keyword}}

\\end{{frontmatter}}

{body}

{bibliography}

\\end{{document}}
"""
    with open('elsarticle/elsarticle-template-num.tex', 'w', encoding='utf-8') as f:
        f.write(final_tex)

if __name__ == '__main__':
    main()
