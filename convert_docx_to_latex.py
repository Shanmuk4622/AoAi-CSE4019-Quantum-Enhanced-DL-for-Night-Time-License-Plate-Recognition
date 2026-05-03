import docx
import re
import os

def escape_latex(text):
    # Basic escaping for LaTeX
    text = text.replace('\\', '\\textbackslash ')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde ')
    text = text.replace('^', '\\textasciicircum ')
    return text

def convert():
    doc_path = r'd:\Documents\V-TOP\Winter-Sem 2025-26\AoAI CSE4019\Project AOAI\Quantum_LPR_Research_Paper.docx'
    doc = docx.Document(doc_path)
    
    out_lines = []
    
    out_lines.append(r'''\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{url}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2026.XXXXXXX}

\title{Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition: A Hybrid Quantum-Classical Neural Network}

\author{\uppercase{Sreenivasa Reddy Edara}\authorrefmark{1}, \uppercase{Shanmukesh Bonala}\authorrefmark{2}}

\address[1]{School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, Andhra Pradesh, India, 522241 (e-mail: sreenivasareddy.e@vitap.ac.in)}
\address[2]{School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, Andhra Pradesh, India, 522241 (e-mail: Shanmukesh.23BCE20070@vitapstudent.ac.in)}

\tfootnote{This work was supported by VIT-AP University.}

\markboth
{Edara and Bonala \headeretal: Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition}
{Edara and Bonala \headeretal: Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition}

\corresp{Corresponding author: Sreenivasa Reddy Edara (e-mail: sreenivasareddy.e@vitap.ac.in).}
''')

    in_abstract = False
    in_keywords = False
    in_list = False
    
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    
    for i, para in enumerate(paragraphs):
        text = para.text.strip()
        style = para.style.name
        
        # Skip the title and author lines at the beginning
        if i < 7:
            if 'Quantum-Enhanced' in text or 'Sreenivasa' in text or 'Corresponding' in text:
                continue
        
        if text == 'Abstract':
            in_abstract = True
            out_lines.append(r'\begin{abstract}')
            continue
            
        if in_abstract:
            if text.startswith('Keywords'):
                out_lines.append(r'\end{abstract}')
                out_lines.append('')
                out_lines.append(r'\begin{keywords}')
                kw_text = text.replace('Keywords—', '').replace('Keywords-', '').replace('Keywords', '').strip()
                out_lines.append(escape_latex(kw_text))
                out_lines.append(r'\end{keywords}')
                out_lines.append('')
                out_lines.append(r'\titlepgskip=-21pt')
                out_lines.append(r'\maketitle')
                out_lines.append('')
                in_abstract = False
                continue
            else:
                out_lines.append(escape_latex(text))
                continue
                
        # Handle regular sections
        if style == 'Heading 1':
            if in_list:
                out_lines.append(r'\end{itemize}')
                in_list = False
            
            # Clean up numbering (e.g., "1. Introduction" -> "Introduction")
            clean_text = re.sub(r'^\d+\.\s*', '', text)
            out_lines.append(f'\\section{{{escape_latex(clean_text)}}}')
            if clean_text == 'Introduction':
                out_lines.append(r'\label{sec:introduction}')
            continue
            
        if style == 'Heading 2':
            if in_list:
                out_lines.append(r'\end{itemize}')
                in_list = False
            clean_text = re.sub(r'^\d+\.\d+\s*', '', text)
            out_lines.append(f'\\subsection{{{escape_latex(clean_text)}}}')
            continue
            
        if style == 'Heading 3':
            if in_list:
                out_lines.append(r'\end{itemize}')
                in_list = False
            clean_text = re.sub(r'^\d+\.\d+\.\d+\s*', '', text)
            out_lines.append(f'\\subsubsection{{{escape_latex(clean_text)}}}')
            continue
            
        if style == 'List Paragraph':
            if not in_list:
                out_lines.append(r'\begin{itemize}')
                in_list = True
            out_lines.append(f'\\item {escape_latex(text)}')
            continue
            
        if style == 'Caption':
            if in_list:
                out_lines.append(r'\end{itemize}')
                in_list = False
            # This is a bit tricky, we'll just put it as a comment for manual fixing
            out_lines.append(f'% CAPTION: {escape_latex(text)}')
            continue
            
        if in_list and style != 'List Paragraph':
            out_lines.append(r'\end{itemize}')
            in_list = False
            
        if style == 'Normal':
            if text.startswith('Table '):
                out_lines.append(f'% TABLE TITLE: {escape_latex(text)}')
            elif text.startswith('['):
                # Probably a reference
                out_lines.append(escape_latex(text))
            elif text.startswith('U(x, '):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'U(x, \theta) = U_L(\theta_L) \cdot U_{L-1}(\theta_{L-1}) \cdot \dots \cdot U_1(\theta_1) \cdot E(x)')
                out_lines.append(r'\end{equation}')
            elif text.startswith('LE_n(x)'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'LE_n(x) = LE_{n-1}(x) + A_n(x) [ LE_{n-1}(x)^2 - LE_{n-1}(x) ]')
                out_lines.append(r'\end{equation}')
            elif text.startswith('f(x, '):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'f(x, \theta) = \langle \psi(x, \theta) | O | \psi(x, \theta) \rangle')
                out_lines.append(r'\end{equation}')
            elif text.startswith('∂f/∂'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'\frac{\partial f}{\partial \theta_k} = \frac{1}{2} \left[ f\left(\theta_k + \frac{\pi}{2}\right) - f\left(\theta_k - \frac{\pi}{2}\right) \right]')
                out_lines.append(r'\end{equation}')
            elif text.startswith('|ψ_0'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'|\psi_0(x)\rangle = \bigotimes_{i=0}^{7} R_X(x_i) |0\rangle')
                out_lines.append(r'\end{equation}')
            elif text.startswith('|ψ('):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'|\psi(x, \theta)\rangle = U_{SEL}(\theta_2) \cdot U_{SEL}(\theta_1) \cdot |\psi_0(x)\rangle')
                out_lines.append(r'\end{equation}')
            elif text.startswith('y_i = 〈ψ'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'y_i = \langle \psi(x, \theta) | Z_i | \psi(x, \theta) \rangle \in [-1, +1], \quad i = 0, \dots, 7')
                out_lines.append(r'\end{equation}')
            elif text.startswith('N_quantum'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'N_{quantum} = L \times n \times 3 = 2 \times 8 \times 3 = 48')
                out_lines.append(r'\end{equation}')
            elif text.startswith('y = W_2'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'y = W_2 \cdot \tanh(W_1 \cdot x + b_1) + b_2, \quad W_1 \in \mathbb{R}^{16\times8}, W_2 \in \mathbb{R}^{8\times16}')
                out_lines.append(r'\end{equation}')
            elif text.startswith('L_CTC'):
                out_lines.append(r'\begin{equation}')
                out_lines.append(r'L_{CTC}(y, l) = - \log \sum_{\pi \in B^{-1}(l)} \prod_{t=1}^{T} y_{\pi_t, t}')
                out_lines.append(r'\end{equation}')
            else:
                out_lines.append(escape_latex(text))
                
        out_lines.append('')
        
    out_lines.append(r'''\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author1.png}}]{Sreenivasa Reddy Edara}
Prof. Edara Sreenivasa Reddy is a distinguished academician, researcher, and administrator currently serving as a Senior Professor (HAG) in the School of Computer Science and Engineering (SCOPE) at VIT-AP University, Amaravati. His illustrious career spans over 32 years in teaching and 16 years in research, complemented by 24 years of administrative leadership.

Prof. Reddy holds a Ph.D. in Computer Science \& Engineering from Acharya Nagarjuna University, an M.Tech. from Visveswaraiah Technological University, and an M.S. in Electronics \& Control from BITS, Pilani. His primary research expertise lies in Machine Learning, Deep Learning, Soft Computing, Image Processing, and Pattern Recognition.

With over 260 publications and a Google Citation h-index of 18, Prof. Reddy is widely recognized for his scientific contributions. He has received numerous honors, including the Best Researcher Award (ANU, 2023), the Best Computer Teacher award (ISTE, 2022), the Sarvepalli Radhakrishnan Pratibha Puraskaram (2020), and the Governor's Gold Medal (2004).
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author2.png}}]{Shanmukesh Bonala}
Shanmukesh Bonala is an AI/ML researcher currently pursuing a B.Tech. degree in Computer Science and Engineering at VIT-AP University, Amaravati, Andhra Pradesh, India. As an aspiring machine learning engineer, he maintains active research interests in trustworthy AI, large language models, neuro-symbolic reasoning, computer vision, and efficient deep learning systems. In addition to his research contributions, he is actively involved in implementing complex algorithmic solutions and exploring the intersections of quantum computing and artificial intelligence.
\end{IEEEbiography}

\EOD

\end{document}
''')

    with open(r'd:\Documents\V-TOP\Winter-Sem 2025-26\AoAI CSE4019\Project AOAI\ACCESS_latex_template_20240429\access_draft.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
        
    print("Draft generated successfully.")

if __name__ == '__main__':
    convert()
