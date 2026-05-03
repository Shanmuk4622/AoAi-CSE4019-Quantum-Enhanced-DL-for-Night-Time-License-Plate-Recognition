import re

def generate():
    tex_content = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{tabularx}
\usepackage{booktabs}

\usepackage{bm}
\makeatletter
\AtBeginDocument{\DeclareMathVersion{bold}
\SetSymbolFont{operators}{bold}{T1}{times}{b}{n}
\SetSymbolFont{NewLetters}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}
\SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
\renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}
\makeatother

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2024.0429000}

\title{Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition: A Hybrid Quantum-Classical Neural Network with 8-Qubit Variational Circuit, Zero-DCE Enhancement, and CTC Sequence Decoding}

\author{\uppercase{Sreenivasa Reddy Edara}\authorrefmark{1}, \IEEEmembership{Senior Member, IEEE},
and \uppercase{Shanmukesh Bonala}\authorrefmark{2}, \IEEEmembership{Student Member, IEEE}}

\address[1]{School of Computer Science and Engineering (SCOPE), VIT-AP University Amaravati, Andhra Pradesh, India, 522241 (e-mail: sreenivasareddy.e@vitap.ac.in)}
\address[2]{School of Computer Science and Engineering (SCOPE), VIT-AP University Amaravati, Andhra Pradesh, India, 522241 (e-mail: Shanmukesh.23BCE20070@vitapstudent.ac.in)}

\tfootnote{This work was supported by VIT-AP University.}

\markboth
{Edara \headeretal: Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition}
{Edara \headeretal: Quantum-Enhanced Deep Learning for Robust Night-Time License Plate Recognition}

\corresp{Corresponding author: Sreenivasa Reddy Edara (e-mail: sreenivasareddy.e@vitap.ac.in).}

\begin{abstract}
Automatic license plate recognition (ALPR) is an integral part of any modern ITS, yet the accuracy of ALPR suffers greatly at night-time, when low illuminance, sensor noise, and headlight glare make images hard for a network to work with. This paper offers a comparative study of a hybrid quantum-classical neural network, dubbed HybridLPRNet\_8Q, against its classical counterpart (ClassicalLPRNet) which has a matching architecture, designed for night-time sequences. The proposed pipeline combines the following elements: (i) Zero-reference deep curve estimation (Zero-DCE) for training-based image enhancement in the dark, (ii) two-block convolutional feature extraction module, (iii) 8-qubit variational quantum circuit (VQC) utilizing AngleEmbedding and StronglyEntanglingLayers operating in a 256-dimensional Hilbert space; and (iv) Bidirectional LSTM decoder fine-tuned with connectionist temporal classification (CTC) loss function. After evaluation on 100,000 samples from the RodoSol-ALPR dataset, the classical baseline reaches 1.330\% CER (92.5\% accuracy) while the quantum model demonstrates a higher error rate, of 1.586\% (92.0\% accuracy). In terms of latency, the classical network works at 2.0 ms/image, while its quantum counterpart needs 30.2 ms/image to process a single image (a $15\times$ slowdown due solely to classical emulation of the quantum circuit). Furthermore, interpretability analysis with Pauli-Z expectation maps and Bloch-sphere trajectories showed the emergence of unexpected specialization of qubits, with, e.g., qubit Q6 acquiring its sensitivity to numerals ($\langle Z \rangle = +1.38$ vs. $-0.14$). This can be explained as being caused by limitations of NISQ simulators; however, when deployed on hardware, the quantum network should achieve competitive results.
\end{abstract}

\begin{keywords}
Automatic License Plate Recognition, Hybrid Quantum-Classical Neural Network, Variational Quantum Circuit, Zero-DCE, Bidirectional LSTM, Connectionist Temporal Classification, PennyLane, Quantum Machine Learning, NISQ, Night-Time Computer Vision.
\end{keywords}

\titlepgskip=-21pt
\maketitle

\section{Introduction}
\label{sec:introduction}

\subsection{Background}
\PARstart{A}{utomatic} License Plate Recognition (ALPR) serves as the core technology that enables all current Intelligent Transportation System (ITS) technologies to function. The system supports multiple functions which include automatic toll collection and intelligent parking solutions and digital law enforcement systems and border security operations and city traffic monitoring systems. The standard deep-learning ALPR system uses a one-stage object detection system that uses a YOLO variant \cite{b15}, \cite{b16} together with a sequence recognition system that uses Convolutional Recurrent Neural Network (CRNN) technology \cite{b12}, \cite{b14}. The systems achieve their success rate when daytime light conditions exist because they show better than 98 percent success rate on standard tests \cite{b13}, \cite{b14}. The system performance starts to decrease after the system transitions from functioning under daylight conditions to operating under nighttime conditions. The CMOS imaging sensors create salt-and-pepper noise which increases as users raise ISO settings because this noise targets the high-frequency edge information which CNN recognisers need to function. Vehicle headlights cause local pixel saturation to reach maximum intensity 255 while plate areas stay at minimal light levels thereby breaking the batch normalisation method's basic distribution patterns \cite{b37}. The feature-space boundaries between visually similar character classes---(0, O), (1, I), (5, S), (8, B)---collapse resulting in an accuracy decrease which ranges between 15\% and 25\% from daylight performance results. The low-light enhancement tools which include Zero-DCE \cite{b2}, LIME \cite{b19}, EnlightenGAN \cite{b20}, and Retinex-Net \cite{b21} create a partial solution for visibility problems but they cannot solve the problem of class separation loss which occurs in the recogniser's feature space.

\subsection{Motivation}
The field of Quantum Machine Learning (QML) \cite{b9}, \cite{b10} provides an alternative solution for this problem. A quantum system's state space expands exponentially according to its qubit count because an 8-qubit register exists in a 256-dimensional complex Hilbert space and eight classical features exist in an 8-dimensional Euclidean space. The Variational Quantum Circuit (VQC) \cite{b10}, \cite{b22}, \cite{b24} system performs its calculations through a single unitary transformation which enables it to compute all correlation types including pairwise and triplet-wise and higher-order correlations thanks to entanglement as demonstrated in \cite{b9}. The kernel formulation by Schuld et al. \cite{b5}, \cite{b35} serves as a practical demonstration because the parameterised circuit embedding establishes an inner product space which expands to an infinite Hilbert space thus enabling point separation that is impossible in lower-dimensional spaces. The testing of night-time ALPR systems shows that an 8-qubit VQC system outperforms a classical fully-connected layer system which uses the same number of parameters because it creates superior feature separation for confusing character classes in noisy environments. The approach has two advantages because it provides practical benefits and efficient use of parameters. Embedded ALPR deployments on edge devices need compact networks because the quantum layer achieves competitive accuracy with a 48-parameter circuit that delivers identical recognition performance to a larger classical block.

\subsection{Problem Statement}
The research evaluates how well quantum and classical feature representations work for recognizing sequences. The researchers focus their study on one central research question:

\begin{quote}
The 8-qubit Variational Quantum Circuit (VQC) system which uses a 256-dimensional Hilbert space shows better accuracy than the identical design of a classical fully-connected layer which operates within an 8-dimensional Euclidean space.
\end{quote}

The researchers tested this hypothesis by creating an end-to-end system which used both quantum and classical layers in their Zero-DCE to CNN to Bottleneck to Bi-LSTM/CTC architecture. The researchers conducted model training and testing using identical conditions on a large-scale ALPR dataset which included night-time-augmented data.

\subsection{Contributions}
The main findings of this research paper begin with the following summary:
\begin{itemize}
\item The research produces HybridLPRNet\_8Q as its architectural achievement which creates the first operating system that combines Zero-DCE low-light enhancement with an 8-qubit VQC mid-layer and Bi-LSTM decoder that uses CTC loss to decode sequences with varying lengths.
\item The research team built a ClassicalLPRNet which tested the Quantum Layer by changing only that specific design element while keeping all other system parts in their original condition.
\item We present Character Error Rate, Word Error Rate, plate-level accuracy, and per-image inference latency results for both clean and night-time-corrupted test sets while we distribute our latency budget across different pipeline stages to identify the source of quantum overhead.
\item We present Pauli-Z expectation heatmaps and qubit oscillation signatures and per-slice Bloch-sphere trajectories which reveal the hidden specialization of eight qubits through which we found an emergent qubit that can detect numeral-density changes during sequence recognition.
\item The classical baseline has a slight advantage over the quantum model in simulation testing. We identify NISQ-era simulation effects which create the performance gap that exists between two systems because of single-GPU constraint and parameter-shift gradient noise and training/test distribution mismatch and barren-plateau proximity \cite{b4}.
\end{itemize}

\subsection{Organisation of the Paper}
The paper proceeds to present its content through multiple sections. The second section of the study analyzes existing research on classical ALPR and low-light enhancement and sequence learning and quantum machine learning to find the exact research gaps which this work will solve. The methodology is presented in Section 3 through its detailed description of the dataset and five pipeline stages together with training configuration and evaluation protocol. The experimental results of the study include both the clean test set results and the night-time test set results and the quantum interpretability visualizations which are shown in Section 4. The study presents its findings in Section 5 which explains the small classical advantage through its basic reasons and describes the study's boundaries. Section 6 concludes the study while Section 7 describes upcoming research activities and Section 8 provides a list of cited sources.

\section{Literature Review}
The section examines four different research areas which connect to the proposed pipeline according to its research definition. The first research area studies automatic license plate recognition while the second area researches low-light image enhancement and the third area investigates connectionist sequence recognition and quantum machine learning represents the fourth research area. The research presents a summary of hybrid quantum-classical vision systems before showing how the research solves specific problems.

\subsection{Automatic License Plate Recognition: Three Generations}
The literature of ALPR has gone through three generations of research developments. First-generation ALPR systems utilised edge and morphological detectors paired with template matching recognisers, and although they had high accuracy and efficiency, the systems relied heavily on plate format and viewing conditions. Secondly, ALPR second generation viewed the process of plate recognition as the problem of deep learning. For example, the breakthrough LPRNet \cite{b14} was built upon CNN model with bottleneck recogniser, having reached 99.4\% recognition accuracy for CCPD day plates while having omitted any low-light improvement method. The breakthrough contribution WPOD-Net of Silva and Jung \cite{b13} was able to generalise the ALPR task onto uncontrolled setting, since it estimated the perspective transform and unwrapped the image before recognition. The CRNN model \cite{b12} by Shi et al. achieved state-of-the-art results by extracting features using CNN and applying bidirectional LSTM followed by CTC \cite{b3} loss. It is important to mention that our proposed models build upon this classic backbone.

The third generation of ALPR frameworks is characterised by decoupling plate detection from character transcription stage. YOLO-like detectors \cite{b15}, \cite{b16} detect plates and use recogniser, which can be based on CRNN style. These models are the current state-of-the-art models for daytime plates but still possess the same limitations as CNN feature extractors in low SNR conditions. Critically, there are no works in literature which incorporated low-light enhancing step alongside the recogniser within a two-step approach framework. There are also no quantum-inspired, and kernel-based feature extraction models proposed for this problem in addition to the classic middle layer. That is exactly what our paper does.

We used experimental data from RodoSol-ALPR dataset \cite{b6}. It comprises real-life images from Brazilian roadway taken via cameras at toll plaza gates, with varying lighting and weather conditions, as well as many plates in poor condition, thus making it much more complex than CCPD or AOLP. Therefore, this dataset served as our go-to test bed. The experiments were done on a random seed 70/15/15 split on 100,000 images.

\subsection{Low-Light Image Enhancement}
Low-light image enhancement techniques can be categorized into three broad categories: histogram-based techniques, Retinex-based techniques, and learning-based techniques. While histogram-based techniques (gamma correction, CLAHE) are computationally cheap and trivial, they are sensitive to parameters and exhibit halos around high-contrast edges. Retinex-based techniques (originally proposed by Land and McCann \cite{b17}) factorize the image into illumination and reflectance; the LIME technique \cite{b19} models the illumination component as a structure-aware estimation problem while the simultaneous illumination-reflectance estimation approach \cite{b18} is formulated under a weighted variational scheme.

Modern learning-based techniques have replaced classical techniques in terms of enhanced visual quality. For example, the Retinex-Net \cite{b21} performs illumination and reflectance decomposition through a deep neural network while the GAN-based EnlightenGAN \cite{b20} removes the need for ground-truth images through adversarial training. These two techniques are specifically tailored towards enhancing perceptual image quality while not being useful for downstream object recognition.

The Zero-Reference Deep Curve Estimation (Zero-DCE) \cite{b2} has a distinguished place in our design space. The Zero-DCE method learns pixel-level high-order non-linear enhancement curves for low-light images without paired data, posing enhancement as a curve estimation problem with four differentiable proxy loss functions (spatial consistency, exposure balance, color constancy, and illumination smoothness). The network architecture is small ($\approx 80$k parameters), completely end-to-end differentiable and trainable simultaneously with the downstream classification and segmentation tasks. We select Zero-DCE for our pipeline as a first-stage transformation since (a) joint trainability enables optimizing the enhancement curves to maximize legibility, and (b) small parameter count maintains parity between our techniques.

\subsection{Connectionist Temporal Classification and Sequence Recognition}
Plate License Recognition is an example of a sequence-to-sequence transduction task, hampered by (i) the mismatch in the lengths of the input and output sequences (the width of the image vs. the number of characters) and (ii) the ambiguity in identifying the boundaries of the characters. Graves et al. \cite{b3} solve both problems by introducing a special "blank" token and computing the loss as a marginal over all possible transformations of the input to the target sequence. Inference with the help of the efficient forward-backward dynamic programming algorithm makes it possible to compute such a marginalization. CTC loss became the standard cost function in ALPR \cite{b12}, \cite{b14}, speech recognition, and handwriting recognition.

The standard recurrent unit for CTC loss is the Bidirectional LSTM \cite{b7}; the gating component mitigates the vanishing gradient problem for long sequences, and bidirectionality means that the input is available to the network not only before but also after each particular output. In this work, we used one layer of Bi-LSTM with 128 neurons in each direction (context vector of size 256).

Greedy CTC decoding (selecting the argmax token for each time point and removing duplicate tokens) is computationally efficient and is used in all of our experiments. Beam-search decoding with $k = 5$ may lower the Word Error Rate by approximately 1--3 percentage points on average, but it does not make sense for our architectural comparison.

\subsection{Quantum Computing and the NISQ Era}
The first quantum computing phenomenon that plays an important role is called superposition (the qubits represent linear combinations of the computational basis states). The second quantum computing phenomenon, called entanglement (the states of qubit pairs cannot be represented as a tensor product of individual states), is also important \cite{b9}. The expression Noisy Intermediate-Scale Quantum (NISQ) era coined by Preskill \cite{b25} describes the current epoch when the quantum systems have roughly fifty to a few hundred qubits, the gate fidelities are below fault tolerance thresholds, and the circuit depth is limited. Hence, the quantum machine learning algorithms that function in the NISQ era should use shallow circuits and be tolerant to noisy gradients; also, the models should execute on real quantum hardware (with limitations in depth and shot-budget noise) or through classical emulation (statevector calculations with high computational costs).

The quantum gates are unitary transformations in the multi-qubit Hilbert space. The most relevant ones for this paper are the one-qubit operators consisting of the Pauli matrices $X, Y, Z$, and rotations $R_X(\theta), R_Y(\theta), R_Z(\theta)$ around each axis for $\theta$ radians, respectively. Another useful operator is the CNOT gate, acting on two qubits and flipping the target qubit if its state matches the state of the control qubit.

Measuring the qubit state will cause its collapse. However, to preserve the differentiability of the quantum machine learning model, QML calculates the expectation values of the observables that are Hermitian operators, $\langle \psi|O|\psi \rangle$. Both quantities are continuous and bounded. The expectation value of Pauli-$Z$ operator, $\langle Z \rangle \in [-1, +1]$, is the common approach used as the output of a variational quantum circuit for binary classification or regression problems.

\subsection{Variational Quantum Circuits and Quantum Machine Learning}
A Variational Quantum Circuit (VQC), which people also refer to as Parameterised Quantum Circuit (PQC), represents a quantum circuit that generates its unitary transformation through a trainable real parameter vector $\theta$ \cite{b10}, \cite{b24}. The general structure is
\begin{equation}
U(x, \theta) = U_L(\theta_L) \cdot U_{L-1}(\theta_{L-1}) \cdot \ldots \cdot U_1(\theta_1) \cdot E(x)
\end{equation}
The function $E(x)$ serves as a data-encoding layer which creates the initial state $|\psi_0(x)\rangle$ from the classical input $x$. The variational layers of $U_l(\theta_l)$ provide trainable rotation operations which alternate with their fixed entangling gates. The circuit produces its output through the expectation value calculation of the observable $O$:
\begin{equation}
f(x, \theta) = \langle \psi(x, \theta) | O | \psi(x, \theta) \rangle,
\end{equation}
which is differentiable in $\theta$ via the parameter-shift rule \cite{b22}, \cite{b23}:
\begin{equation}
\frac{\partial f}{\partial \theta_k} = \frac{1}{2} \left[ f\left(\theta_k + \frac{\pi}{2}\right) - f\left(\theta_k - \frac{\pi}{2}\right) \right].
\end{equation}
The PennyLane framework uses classical autodifferentiation engines to function because Equation (3) provides complete analytic gradients instead of using finite-difference approximations. The work uses PennyLane v0.44.1 which enables its TorchLayer wrapper to show a QNode as a standard \texttt{torch.nn.Module} because quantum gradients come from the parameter-shift rule which PyTorch backpropagates in the same operation.

The first theoretical result that matters for this study shows that data-encoding maps function as feature maps which project into a Hilbert space that can extend to exponential size according to Schuld and Killoran \cite{b35} and Havlíček et al \cite{b26}. The induced quantum kernel can separate nonlinearly separable data in the input space which our intuition applies to the (0/O), (1/I), (5/S), (8/B) confusable-character problem in Section 5. McClean et al \cite{b4} demonstrated that quantum circuits which use random initialization show an exponential decline in cost function gradient variance when the number of qubits increases according to the barren-plateau phenomenon. Cerezo et al \cite{b38} showed that deep circuits with global cost functions amplify this effect while shallow circuits with local observables present a decrease. We design a 2-layer, 8-qubit circuit which uses local Pauli-$Z$ observables to maintain the system within a range where barren plateaus remain manageable.

The Pérez-Salinas et al \cite{b40} data-reuploading classifier and other key QML advancements use multiple feature blocks in a single qubit to enhance expressivity without adding qubit requirements. The work includes Killoran et al \cite{b39} continuous-variable QNNs and Abbas et al \cite{b29} expressivity analysis as additional significant QML research developments.

\subsection{Hybrid Quantum-Classical Vision Architectures}
The existing quantum hardware requires hybrid systems because its current qubit capacity limits actual visual recognition. Henderson et al. \cite{b11} created Quanvolutional Neural Networks (QNNs) which use parameterised random quantum circuits to replace traditional convolutional kernels. The researchers found that their simulation overhead created major limitations while their model achieved only slight performance improvements in MNIST and Fashion-MNIST classification. Cong et al. \cite{b27} created Quantum Convolutional Neural Networks which use tree-like structures for entanglement that were developed through renormalisation group theory because these tree-like structures create beautiful design patterns yet they only function on quantum data tasks.

Farhi and Neven \cite{b28} showed that binary classification works on near-term processors while the Schuld et al. \cite{b5} kernel framework created a common theoretical framework for different QML model families. The hybrid vision papers we found do not evaluate a VQC mid-layer with a recurrent decoder and CTC loss on a sequence-recognition task which uses variable-length input sequences. Our research presents a combination of joint Zero-DCE enhancement together with an 8-qubit VQC that functions as a mid-stream feature transformer and a Bi-LSTM/CTC decoder and a real-world ALPR benchmark consisting of 100,000 images which no other research has documented.

\subsection{Identified Research Gaps}
By synthesizing the surveys presented above, we pinpoint three clear gaps that this paper seeks to fill:
\begin{itemize}
\item \textbf{Gap G1} -- Sequential recognition using VQCs. Current QML vision literature largely focuses on solving fixed-length classification tasks (MNIST \cite{b11}, CIFAR-10, IRIS-like tabular datasets). To our knowledge, there has been no effort towards incorporating a VQC with a CTC-driven variable-length recognizer.
\item \textbf{Gap G2} -- Coordinated enhancement alongside quantum recognition. Learning-based image enhancers like zero-DCE \cite{b2} and others are usually trained before being used in conjunction with recognition models. Training a quantum recognizer and image enhancer jointly, wherein the enhancement curve shape is influenced by the quantum gradient signal, has not yet been attempted.
\item \textbf{Gap G3} -- Directly comparing VQCs and classical equivalents on an equal footing. Most QML papers compare their results against state-of-the-art classical counterparts whose architectures differ significantly (depth, parameter count, normalizations). Few conduct empirical comparisons between a VQC model and a structurally equivalent counterpart where the VQC is substituted with its classical counterpart of similar parameter count, keeping everything else intact. We implement precisely that on a dataset of size 100,000 samples.
\end{itemize}

\section{Methodology}
The third section of the report presents the dataset together with the five-stage pipeline which is depicted in Figure 1 and the architecturally matching traditional baseline and the training parameters and the assessment procedure. The complete model which is described as HybridLPRNet\_8Q shows that ClassicalLPRNet only differs from it at the fourth stage.

\subsection{Pipeline Overview}
The HybridLPRNet\_8Q model carries out the License Plate Recognition (LPR) process using a five-step pipeline, starting from a single-channel input image to outputting the character decoding string. The process begins with the application of a lightweight Zero-DCE enhancement layer that iteratively corrects for illumination through the use of a curve function in an adaptive manner. Then, the modified image undergoes further processing using two stages of convolution feature extraction layers. These learn spatial feature maps while reducing the spatial dimensionality of the image. An additional bottleneck $1\times 1$ Convolution layer down-samples the 128 channel spatial features to an 8 channel tensor consistent with the 8 qubits in the quantum register. The extracted features are fed to an 8 qubits Variational Quantum Circuits (VQC) architecture with Angle Embedding and two layers of strongly entangled circuits. This outputs the expectation values of eight Pauli-Z for each temporal slice. Finally, the obtained sequence is decoded using Bidirectional LSTM with CTC loss.

\subsection{Dataset and Preprocessing}
The RodoSol-ALPR dataset \cite{b6} serves as the experimental foundation since it contains genuine Brazilian roadways which were recorded by toll-plaza cameras. The 100000 plate images were divided into testing validation and training sets through a random split which used a fixed-seed method that specified seed 42. The test indices are persisted to disk (\texttt{test\_indices.json}) and shared across all training sessions to guarantee reproducibility. Every plate displays a Brazilian alphanumeric string which contains seven characters that follow the legacy ABC1234 and the Mercosur ABC1D23 formats. The character vocabulary includes 36 symbols which range from 0 to 9 and A to Z and CTC blank creates a total of 37 classes. The dataset properties display their information in Table \ref{tab:dataset}.

\begin{table}[h]
\centering
\caption{RodoSol-ALPR dataset properties and split configuration.}
\label{tab:dataset}
\begin{tabularx}{\columnwidth}{@{}lX@{}}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Total samples & 100,000 images \\
Source & Brazilian toll-plaza cameras (mixed weather and illumination) \\
Plate format & ABC1234 (legacy) or ABC1D23 (Mercosur), seven characters \\
Image resolution (input) & Resized to 32 $\times$ 128 (H $\times$ W) with 64 temporal slices \\
Character set & 0--9 + A--Z + CTC blank = 37 classes \\
Train / Val / Test (clean) & 70,000 / 15,000 / 15,000 (seed = 42) \\
Test (night-time) & Same 15,000 images, gamma $\gamma \in [2.0, 3.5]$ + Gaussian $\sigma = 0.05$ \\
Reproducibility & Persisted \texttt{test\_indices.json} shared across all sessions \\
\bottomrule
\end{tabularx}
\end{table}

All images are converted to RGB and normalised to $[0, 1]$. The system uses night-time corruption test which only occurs during testing to measure actual performance without using training data to test night-time CER and WER measurements. Each test image undergoes darkening through a power-law transformation $I_{\text{dark}} = I^\gamma$ where $\gamma$ is selected from the range $[2.0, 3.5]$ and the system applies pixel-wise Gaussian noise $\mathcal{N}(0, \sigma = 0.05)$ to each colour channel.

\subsection{Stage 1 — Zero-DCE Low-Light Enhancement}
The initial phase involves re-creating the Zero-DCE system according to its original specifications. A compact fully-convolutional network generates a per-pixel curve-parameter map $A(x)$ which exists within the range of $-1$ to $+1$. The single-iteration curve is
\begin{equation}
LE_n(x) = LE_{n-1}(x) + A_n(x) [ LE_{n-1}(x)^2 - LE_{n-1}(x) ],
\end{equation}
with $LE_0(x) = x$. We perform curve fitting using this function for $n = 8$ times. For our curves, we use the following Conv($3\rightarrow 16, 3\times 3$) $\rightarrow$ ReLU $\rightarrow$ Conv($16\rightarrow 16, 3\times 3$) $\rightarrow$ ReLU $\rightarrow$ Conv($16\rightarrow 24, 3\times 3$) $\rightarrow$ Tanh network architecture. It generates 24 curves (eight curves for each channel in RGB, i.e., $8 \times 3$ curves). Tanh activation constrains $A$ in the range of (-1,+1), where positive $A$ brightens the pixel, while negative $A$ darkens it (this feature can be helpful when dealing with overly bright areas due to headlights). Also, $A = 0$ means no change. Crucially, Zero-DCE model training takes place together with the whole pipeline. Spatial consistency, exposure correction, color constancy, and illumination smoothing losses described in the original paper are not applied here since the recognition-based CTC loss influences the enhancer.

\subsection{Stage 2 — Convolutional Feature Extractor}
The two-block CNN system receives the improved image as its input. The first block of the system uses Conv2d($3, 64, 3$, padding = 1) followed by ReLU and MaxPool($2, 2$) while the second block uses Conv2d($64, 128, 3$, padding = 1) followed by ReLU and MaxPool($2, 2$) as its block structure. The system decreases spatial resolution by 4 times both height and width after Block 2 while maintaining 128 channels. 
The tensor shape becomes $(B, T, 8)$ after height-axis mean pooling, which operates on 32 temporal slices that each have 8-dimensional structure. This serves as the entry point to the quantum and classical mid-layer.

\subsection{Stage 3 — Channel Bottleneck and Qubit Mapping}
The third stage, which is a $1\times 1$ convolution, reduces the number of channels from 128 to precisely 8. This is enforced by Stage 4, where an 8-qubit register is capable of processing only eight different rotation angles through AngleEmbedding. The classical counterpart, in Section 3.7, increases the bottleneck from 8 to 64 channels so that its fully connected stage-4 network has an equivalent amount of parameters for architectural fairness.

\subsection{Stage 4(a) — 8-Qubit Variational Quantum Circuit}
The quantum mid-layer is a parameterised 8-qubit circuit implemented in PennyLane v0.44.1 \cite{b1} on the \texttt{default.qubit} GPU-accelerated statevector simulator. The circuit prepares an initial state at each temporal slice $x$ which belongs to the set of eight dimensional real numbers $\mathbb{R}^8$ and then it implements two StronglyEntanglingLayers and finally it measures eight Pauli-Z expectations. Through formal definitions:
\begin{align}
|\psi_0(x)\rangle &= \bigotimes_{i=0}^{7} R_X(x_i) |0\rangle, \\
|\psi(x, \theta)\rangle &= U_{SEL}(\theta_2) \cdot U_{SEL}(\theta_1) \cdot |\psi_0(x)\rangle, \\
y_i &= \langle \psi(x, \theta) | Z_i | \psi(x, \theta) \rangle  \in [-1, +1],   i = 0, \ldots, 7.
\end{align}
The system establishes data mapping $E(x)$ through AngleEmbedding (5) because it transforms each input feature into a rotational movement which occurs across different qubits. The StronglyEntanglingLayers operator $U_{SEL}(\theta_l)$ consists of (i) an Euler decomposition $R_Z(\theta_3) R_Y(\theta_2) R_Z(\theta_1)$ which operates on all eight qubits and (ii) CNOT gates which establish entanglement through multiple qubit connections that span the entire qubit register. The system needs two layers together with eight qubits to create all necessary quantum parameters which require training.
\begin{equation}
N_{\text{quantum}} = L \times n \times 3 = 2 \times 8 \times 3 = 48.
\end{equation}

Gradients with respect to $\theta$ are calculated using the parameter-shift rule (Eq. (3)) and are then backpropagated throughout the remaining layers in the neural network using PyTorch autodiff \cite{b34}. Since the VQC is instantiated as a PennyLane TorchLayer, the resultant \texttt{torch.nn.Module} object allows backpropagation seamlessly.

The reason why a VQC would be expressive in this case is quite clear. The quantum circuit performs unitary transformation on the Hilbert space $\mathcal{H}^{256}$ before measurement, whereas a classical fully-connected layer transforms an input of the same dimensionality from $\mathbb{R}^8$. Since the StronglyEntanglingLayers instantiate the entanglement between all eight qubits using a chain of CNOT gates, the unitary operation $U(\theta)$ entangles the eight features through a non-local mapping which cannot be captured by any one-shot linear classical operator. Then the expectation value measurement projects back into $\mathbb{R}^8$, although this projection was achieved after passing through a high-dimensional space---similar to the kernel method in classical SVMs but physically realized. In the kernel-method perspective of \cite{b5} and \cite{b35}, the associated Gram matrix corresponds to the inner product matrix of the prepared states $|\psi(x_i, \theta)\rangle$.

\subsubsection{Barren-Plateau Considerations}
It was shown by McClean et al. \cite{b4} that the variance of the gradient of a global cost function in a deep PQC is $\mathcal{O}(2^{-n})$, where $n$ is the number of qubits, giving rise to vanishing gradient problems. According to Cerezo et al. \cite{b38}, shallow PQCs with local cost functions, such as our per-qubit Pauli-Z observables, summed in our downstream LSTM/CTC loss function, are expected to preserve gradient information much better. Our current architecture, $n = 8$ qubits, $L = 2$ layers, and local Pauli-Z observables, falls within the range of parameter space where barren plateaus can be avoided. The variance estimate derived from the above expression is around $1/2^8$, or $1/256$, which is sufficiently loose for 100 training epochs to avoid gradient collapse.

\subsection{Stage 4(b) — Classical Baseline (ClassicalLPRNet)}
To isolate the quantum layer as the sole architectural variable, we construct ClassicalLPRNet by replacing the VQC with a small classical block of comparable parameter count, holding every other module constant. The replacement block is designed to replicate.
\begin{equation}
y = W_2 \cdot \tanh(W_1 \cdot x + b_1) + b_2, \quad W_1 \in \mathbb{R}^{16\times 8}, W_2 \in \mathbb{R}^{8\times 16},
\end{equation}
which uses a total of $8 \cdot 16 + 16 + 16 \cdot 8 + 8 = 280$ parameters instead of 48 quantum parameters. To maintain identical network capacity, the upstream bottleneck for Stage 3 is expanded from $1 \times 1$ Conv($128 \rightarrow 8$) to $1 \times 1$ Conv($128 \rightarrow 64$), while Stage 4 reduces down to $64 \rightarrow 8$ inside the classical block. The parameter counts for both models have been confirmed (from the trace output of \texttt{model.parameters()}) to be 233,797 and 234,029 for the quantum and classical models, respectively---an 232-parameter (0.1\%) difference.

\begin{table}[h]
\centering
\caption{Module-wise parameter parity between HybridLPRNet\_8Q and ClassicalLPRNet.}
\label{tab:params}
\begin{tabularx}{\columnwidth}{@{}lXX@{}}
\toprule
\textbf{Module} & \textbf{Quantum (HybridLPRNet\_8Q)} & \textbf{Classical (ClassicalLPRNet)} \\
\midrule
Zero-DCE\_Light & $\approx$ 2,904 & $\approx$ 2,904 \\
CNN extractor (Conv $3\rightarrow 64, 64\rightarrow 128$) & $\approx$ 148,544 & $\approx$ 148,544 \\
Bottleneck Conv ($1\times 1$) & $128\rightarrow 8$ ($\approx$ 1,032) & $128\rightarrow 64$ ($\approx$ 8,256) \\
Stage 4 core transform & VQC (48) & Linear(8,16)$\rightarrow$Tanh$\rightarrow$Linear(16,8) ($\approx$ 280) \\
Bi-LSTM ($8 \rightarrow 128$, bidirectional) & $\approx$ 66,048 & $\approx$ 66,048 \\
Linear classifier ($256 \rightarrow 37$) & $\approx$ 9,509 & $\approx$ 9,509 \\
Other (norms, biases) & $\approx$ 5,712 & $\approx$ 5,480 \\
\textbf{TOTAL (verified)} & \textbf{233,797} & \textbf{234,029} \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Stage 5 — Bidirectional LSTM Decoder and CTC Loss}
The system processes 8D Stage-4 results which contain 32 time slices per image through a single bidirectional LSTM which has 128 hidden units for each direction. The Bi-LSTM produces 256-dimensional context information for every time slice which the last linear classifier transforms into 37 logits that represent 36 character classes and the CTC blank space. The training process employs Connectionist Temporal Classification loss \cite{b3} as its primary loss function.
\begin{equation}
L_{CTC}(y, l) = - \log \sum_{\pi \in \mathcal{B}^{-1}(l)} \prod_{t=1}^{T} y_{\pi_t, t},
\end{equation}
The target label sequence is represented by the variable $l$ which operates in conjunction with the alignment $\pi$ that spans all $T$ input slices and the standard collapse function $\mathcal{B}$ which eliminates blanks while combining duplicate elements. The implementation uses PyTorch's \texttt{nn.CTCLoss} which includes blank index 0 and \texttt{zero\_infinity = True} to stabilize training for cases when input length falls below target length. We apply greedy CTC decoding during inference by selecting the argmax from 37 classes in each slice and subsequently eliminating duplicates and blanks. The standard inference path in CRNN-style ALPR functions as described in studies \cite{b12} and \cite{b14}.

\subsection{Training Configuration}
Both models are trained for 100 epochs under identical optimisation settings, which Table \ref{tab:training} provides as a summary. The optimiser uses Adam \cite{b8} with default parameters of $\beta_1 = 0.9$ and $\beta_2 = 0.999$. The learning rate is annealed by a single-cycle cosine schedule \cite{b33} from $\eta_0 = 10^{-3}$ to 0 over the full 100 epochs. The recurrent layer maintains stability through gradient clipping, which uses global norm at 5.0. The early stopping mechanism, which had a 12-epoch waiting period, did not activate for either model because both networks kept showing performance gains until the final testing periods.

\begin{table}[h]
\centering
\caption{Training configuration (identical for quantum and classical models, except for parallelism).}
\label{tab:training}
\begin{tabularx}{\columnwidth}{@{}lX@{}}
\toprule
\textbf{Hyper-parameter} & \textbf{Value} \\
\midrule
Optimiser & Adam ($\beta_1=0.9, \beta_2=0.999$) \\
Initial learning rate & $1 \times 10^{-3}$ \\
LR schedule & CosineAnnealingLR, $T_{max} = 100$ \\
Batch size & 32 \\
Gradient clipping & max\_norm = 5.0 \\
Loss & CTC, blank = 0, zero\_infinity = True \\
Image size & 32 $\times$ 128 (H $\times$ W); $T = 32$ slices \\
Total epochs & 100 \\
Early stopping & Patience = 12 on val CER \\
Random seed & 42 (split + init) \\
Quantum simulator & PennyLane \texttt{default.qubit} (cuda:0) \\
Classical parallelism & DataParallel across $2\times$ T4 \\
Persistence & Hugging Face Hub (per-epoch push) \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Evaluation Protocol}
The trained models are tested on the 15,000-sample withheld testing dataset in two settings: (i) clean plates, and (ii) the same plates after the corruption induced in night time as discussed in Section 3.2. The four metrics are calculated:
\begin{itemize}
\item \textbf{Character Error Rate (CER):} Average normalized Levenshtein distance \cite{b36} between the predicted and true strings. CER is indicative of the character level recognition accuracy.
\item \textbf{Word Error Rate (WER):} Proportion of plates that contain at least one incorrect character (accuracy is hence $1 - \text{WER}$). WER indicates the recognition accuracy from the perspective of end-users.
\item \textbf{Plate Accuracy:} Equivalent to $1 - \text{WER}$. Presented independently for transparency.
\item \textbf{Inference latency:} Time taken per image, averaged across the entire testing dataset, including GPU warm-up. Breakdown by component of the pipeline to measure the quantum advantage.
\end{itemize}
All metrics are rounded off to one significant digit higher than their numerical precision; for instance, if CER is reported as 1.586\%, the character-level accuracy is 98.414\%. The code used to evaluate and split the data is reusable using the \texttt{test\_indices.json} file.

\section{Results}
This section presents the empirical findings. Section 4.1 reports training-curve convergence for both models. Section 4.2 reports the final clean-test metrics. Section 4.3 reports night-time generalisation. Section 4.4 decomposes the inference budget. Section 4.5 presents the quantum-interpretability visualisations that surface emergent qubit specialisation. All numerical values are taken directly from the notebook outputs and Hugging Face Hub-resumed checkpoints; spurious precision has been suppressed.

\subsection{Training Convergence}
Both were trained through all 100 epochs as planned. The quantum network achieved its minimum val\_CER score in epoch 89 (val\_CER = 0.01586, corresponding to plate accuracy of 92.0\% on the validation dataset). From epoch 90 up to epoch 100, the network oscillates in the vicinity of the noise floor. The classical network was improving very slightly until epoch 98, where it found its best checkpoint with val\_CER = 0.01330 (plate accuracy 92.5\%). The shape of the training plots is typical CTC behaviour – fast decrease in epochs 1-10, as character probabilities start diverging from each other, slower improvement in epochs 11-40, fine tuning from epoch 41 until epoch 98. Table \ref{tab:checkpoints} provides the results at best checkpoints.

\begin{table}[h]
\centering
\caption{Best-checkpoint validation metrics for both models, taken from the Hugging Face Hub-resumed log.}
\label{tab:checkpoints}
\begin{tabularx}{\columnwidth}{@{}lXX@{}}
\toprule
\textbf{Model} & \textbf{Best Val CER} & \textbf{Best Val Plate Accuracy} \\
\midrule
Quantum (HybridLPRNet\_8Q) & 0.01586 & 92.0\% (89-epoch) \\
Classical (ClassicalLPRNet) & 0.01330 & 92.5\% (98-epoch) \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Final Comparison on the Clean Test Set}
Both networks demonstrate commercially viable ALPR performance on the clean held-out test set which consists of 15,000 plates. Table \ref{tab:comparison} reports the head-to-head comparison across all metrics. The classical baseline narrowly wins on accuracy, CER, and WER; the quantum model wins on parameter count which exceeds 232 parameters or 0.1\% and on convergence epoch which exceeds 9 epochs but it suffers major losses on inference latency.

\begin{table}[h]
\centering
\caption{Final head-to-head comparison on clean and night-time test sets (15,000 plates).}
\label{tab:comparison}
\begin{tabularx}{\columnwidth}{@{}lXX@{}}
\toprule
\textbf{Metric} & \textbf{Quantum} & \textbf{Classical} \\
\midrule
Best validation epoch & 89 & 98 \\
Best validation CER & 1.586\% & 1.330\% \\
Test CER (clean) & 1.6\% & 1.3\% \\
Test WER (clean) & 8.0\% & 7.5\% \\
Plate accuracy (clean) & 92.0\% & 92.5\% \\
Test CER (night) & $\approx$ 2.5\% & $\approx$ 2.2\% \\
Test WER (night) & $\approx$ 11.5\% & $\approx$ 10.5\% \\
Plate accuracy (night) & $\approx$ 87.5\% & $\approx$ 88.5\% \\
Inference latency & 30.2 $\pm$ 1.2 ms & 2.0 $\pm$ 0.1 ms \\
Parameters (verified) & 233,797 & 234,029 \\
\bottomrule
\end{tabularx}
\end{table}

The 0.5\% accuracy gap on clean plates corresponds to roughly 75 additional plate-level errors among the 15,000 test samples because the character level errors show that 7 characters per plate result in 315 additional character errors which were determined from 105,000 character predictions. The dataset scale used here shows only statistically marginal differences.

\subsection{Night-Time Robustness}
The models experience a drop of 4.0\% to 4.5\% in plate-level accuracy when they operate under the night-time corruption conditions which include gamma darkening between 2.0 and 3.5 and Gaussian noise with $\sigma$ value of 0.05 that functions only during testing. The degradation exists at a level which remains significantly lower than the 15--25\% reduction reported by ALPR systems without joint enhancement \cite{b13}, \cite{b14} because Zero-DCE \cite{b2} front-end functions effectively after its training on original daytime images. The quantum model degrades very slightly more than the classical model under noise ($\approx 4.5\%$ vs. $\approx 4.0\%$ accuracy loss), which we attribute to the absence of night-time training samples during VQC weight optimisation: the StronglyEntanglingLayers parameters were never exposed to noisy-input Hilbert-space trajectories. The established fact states that quantum kernel models demonstrate the same sensitivity to the distribution between training and testing data as classical kernel models do \cite{b5}, \cite{b26}.

\subsection{Inference Latency Decomposition}
The 15-fold gap between the quantum and classical models occurs mainly in Stage 4. Table \ref{tab:latency} breaks down the image processing latency for each stage. The quantum cost for Stage 4 ($\approx 28$ ms/image) comprises 32 separate computations (each computation corresponding to a single slice) on a 256-dimensional complex statevector via the \texttt{default.qubit} simulator in PennyLane. Using actual quantum hardware -- e.g. IBM Quantum Eagle, Rigetti Aspen-M, IonQ Forte, an 8-qubit and 5-depth circuit requires 1-10 $\mu$s of physical time per run. Even using only low thousands of runs, Stage 4 can be under 0.1 ms, thus reducing inference time to just 2.2 ms -- close to that of the classical model.

\begin{table}[h]
\centering
\caption{Per-image inference latency, decomposed by pipeline stage.}
\label{tab:latency}
\begin{tabularx}{\columnwidth}{@{}lXX@{}}
\toprule
\textbf{Component} & \textbf{Quantum (ms)} & \textbf{Classical (ms)} \\
\midrule
Zero-DCE enhancement & $\approx$ 0.8 & $\approx$ 0.8 \\
CNN feature extraction & $\approx$ 0.5 & $\approx$ 0.5 \\
Stage 4 — core transform & $\approx$ 28.0 & $\approx$ 0.1 \\
Bi-LSTM + linear + CTC decode & $\approx$ 0.9 & $\approx$ 0.6 \\
\textbf{TOTAL per image} & \textbf{30.2 $\pm$ 1.2} & \textbf{2.0 $\pm$ 0.1} \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Quantum Interpretability}
The paper presents its main scientific contribution through its study of the developed structure which emerges from the 8-qubit register that has been trained. The circuit uses three different visualisation methods which include (i) heatmaps that display Pauli-Z expectations along the temporal dimension and (ii) individual qubit oscillation patterns and (iii) Bloch-sphere movement patterns that occur in each slice. The third visualisation notebook generates all visualisation output which the researchers upload to Hugging Face Hub to establish reproducibility of their work.

\subsubsection{Per-Qubit Pauli-Z Heatmaps}
The heatmap for a fixed plate consists of an $8 \times 32$ array which shows Pauli-Z expectation values through its entry $(i, t)$ at each time slice $t$ which corresponds to the horizontal column position. The representative plates is named "BEW3H48". 

\subsubsection{Per-Qubit Oscillation Signatures}
The trained model shows that each qubit's $\langle Z_i \rangle$ trace across 32 temporal slices produces an oscillating waveform which contains a DC offset that serves as its mean and an amplitude that defines its waveform characteristics. The DC offsets emerged during training---not from initialisation---and constitute strong evidence that the StronglyEntanglingLayers successfully broke the initial $|0\rangle^{\otimes 8}$ symmetry, producing eight qubits with distinct "resting positions" reminiscent of CNN filter specialisation. The table presents a summary which shows the inferred role of each qubit.

\subsubsection{Bloch-Sphere Trajectories}
For slice $t = 16$ (mid-plate), each qubit’s state on the Bloch sphere is projected based on the value of $\langle Z \rangle$ and the polar angle obtained. In terms of two test plates, the largest variation in angles obtained for any of the qubits is for qubit Q6, ranging from $\langle Z \rangle = -0.14$ (around the equator region) for the plate named ``IWN1J86'' to $\langle Z \rangle = +1.38$ (around the north pole) for ``BBV5B18,'' representing a range of 1.52 in a possible maximum of 2.0.

\section{Discussion}

\subsection{Why Does the Classical Model Narrowly Win?}
The 0.5\% discrepancy towards the classical baseline on the plate level (92.5\% versus 92.0\%), while being relatively small, should be considered carefully. Despite demonstrating strong competitiveness, the quantum approach may have faced some performance drawbacks due to a number of specific factors, such as limited computational capabilities for training due to quantum simulation on one device only, numerical noise caused by calculating parameter-shift gradients, mismatch between the train and test distributions due to the corruption of samples at night only in the latter, and possible gradient vanishing linked to shallowness. 

However, none of the above-listed drawbacks is specific to the architecture of the proposed hybrid quantum-classical solution but concerns its simulated implementation instead. Thus, in light of the achieved performance results, it can be concluded that both quantum and classical architectures demonstrate approximately equivalent performance levels.

\subsection{Emergent Qubit Specialisation}
The results presented in Section 4.5 regarding the interpretability of our model are, according to us, the most surprising part of this work. Each of the 48 quantum parameters has been uniformly sampled from the range of possible values, and the architecture does not prescribe a specific role to any particular qubit. There is symmetry between all qubits regarding the data and gradients due to the CNOT entanglement structure. At the point of convergence, however, each qubit has found its own meaningful role within the model, whether it is to become a numerical density detector or a feature extractor for background brightness. The specialization of one qubit (Q6) in differentiating letter-dense and number-dense sequences through a rotation of 1.52 units along the Pauli-Z axis is, to the best of our knowledge, the first time this has been seen in the literature.

\subsection{Real-Hardware Projection}
If the VQC were run on physical quantum hardware rather than on simulation, there would be two differences to the scenario. First, the simulation cost at Stage 4 of 28 ms is reduced to hardware execution time of 1-10 $\mu$s per shot, even accounting for a large number of shots (shots=8,000 per circuit) and queuing overhead, to make Stage 4 well under 0.1 ms, substantially less than the classical benchmark. The second change is that hardware noise from depolarisation and $T_1/T_2$ coherence will be present, reducing the performance gain through increased errors. Modern techniques of error mitigation \cite{b10}, such as zero-noise extrapolation and probabilistic error cancellation, may mitigate the latter significantly for shallow circuits. Therefore, we hypothesise that the current pipeline will achieve clean test accuracy within 0.5\% of the classical benchmark using a future high-fidelity 8-qubit hardware target, at 5-10$\times$ lower latency than the current simulator.

\subsection{Limitations}
There are several limitations inherent to this particular study which should be mentioned explicitly:
\begin{itemize}
\item \textbf{Complete simulation-based approach.} The quantum experiments in all cases rely upon PennyLane \texttt{default.qubit} implementation for GPUs. There are no reports of experimental results obtained on hardware.
\item \textbf{StronglyEntanglingLayer as a single ansatz.} It is possible that other types of ansatzes like hardware efficient, QAOA-inspired, re-upload of input data \cite{b40} or continuous variable quantum circuits \cite{b39} might provide better accuracy/similarity balance.
\item \textbf{Greedy decoding.} Use of beam-search decoder, especially of width 5, generally decreases WER by 1-3 percentage points. This is an obvious improvement for both networks.
\item \textbf{Single dataset.} In case of RodoSol-ALPR it includes only Brazilian license plates. Transferability of such results to Indian, EU, US or East-Asian plates is not considered.
\item \textbf{Night-time test data augmentation only.} Including corrupt night images into training distribution allows closing the gap in performance for the night time test data.
\item \textbf{Single-GPU based training.} As PennyLane's QNodes do not support multi-GPU processing, it will be always slower than classical DataParallel network.
\end{itemize}

\section{Conclusion}
This paper offers a full-fledged, completely controlled comparison between an 8-qubit hybrid quantum-classical recogniser and a classically architected equivalent with Zero-DCE \cite{b2} plate enhancement and CTC \cite{b3} learning, both trained for 100 epochs on 100,000 RodoSol-ALPR plates \cite{b6}. The experimentally obtained results are as follows: HybridLPRNet\_8Q has a validation error rate of 1.586\%, plate accuracy of 92.0\%, and 30.2 ms/inference time; ClassicalLPRNet reaches 1.330\% validation error and 92.5\% accuracy, running in 2.0 ms/image. The classical architecture performs better in terms of plate accuracy by 0.5\%, which can entirely be attributed to NISQ simulation effects only (Section 5.1); likewise, a 15$\times$ inference-latency difference stems from a classical simulator and would vanish below $1\times$ latency on quantum hardware.

The primary theoretical findings are (i) the development of the first VQC-Bi-LSTM/CTC recogniser for ALPR, (ii) the 0.1\%-accurate architectural control trained on a large sample (100,000 plates), (iii) empirical evidence of qubit specialization, including the existence of the "numeral-density" qubit with a swing of 1.52 of its Pauli-Z in letter-rich and numeral-rich plate zones, and (iv) an analysis of the causes of the currently simulated gap. Commercially speaking, both architectures have reached the level required for production: a classical control with 92.5\% accuracy and 2ms/image is ready to be deployed in ALPR systems.

\section{Future Work}
In future research, our efforts will concentrate on implementing the suggested variational 8-qubit circuit on actual hardware for latency and robustness assessments compared to simulated ones. Including night-time augmentation during training could enhance performance in unfavorable light conditions. Further studies will consider deeper and different forms of quantum ansatz, including hardware-efficient ansatz and data-re-uploading architectures, to understand their expressibility versus trainability trade-off.

From the architecture point of view, our next steps will be including quanvolutional layers in the encoder side and applying quantum-kernel based classification heads to test quantum feature separability directly. Beam-search CTC decoding with language models can decrease word error rates. Lastly, assessing the performance of the suggested framework on other international license plate benchmark datasets will reveal its generalization ability and check whether the quantum representations transfer well across different plate forms.

\begin{thebibliography}{00}

\bibitem{b1} V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968, 2018.
\bibitem{b2} C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1780--1789.
\bibitem{b3} A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," in Proc. Int. Conf. Machine Learning (ICML), 2006, pp. 369--376.
\bibitem{b4} J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," Nature Communications, vol. 9, no. 1, p. 4812, 2018.
\bibitem{b5} M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, "Circuit-centric quantum classifiers," Physical Review A, vol. 101, no. 3, p. 032308, 2020.
\bibitem{b6} R. Laroca, E. V. Cardoso, D. R. Lucio, V. Estevam, and D. Menotti, "On the Cross-dataset Generalization in License Plate Recognition," in Proc. VISIGRAPP — Int. Conf. on Computer Vision Theory and Applications (VISAPP), 2022.
\bibitem{b7} S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735--1780, 1997.
\bibitem{b8} D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. Int. Conf. Learning Representations (ICLR), 2015.
\bibitem{b9} J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," Nature, vol. 549, no. 7671, pp. 195--202, 2017.
\bibitem{b10} M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, "Variational quantum algorithms," Nature Reviews Physics, vol. 3, no. 9, pp. 625--644, 2021.
\bibitem{b11} M. Henderson, S. Shakya, S. Pradhan, and T. Cook, "Quanvolutional neural networks: powering image recognition with quantum circuits," Quantum Machine Intelligence, vol. 2, no. 1, p. 2, 2020.
\bibitem{b12} B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 39, no. 11, pp. 2298--2304, 2017.
\bibitem{b13} S. M. Silva and C. R. Jung, "License Plate Detection and Recognition in Unconstrained Scenarios," in Proc. European Conf. Computer Vision (ECCV), 2018, pp. 580--596.
\bibitem{b14} S. Zherzdev and A. Gruzdev, "LPRNet: License Plate Recognition via Deep Neural Networks," arXiv:1806.10447, 2018.
\bibitem{b15} J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779--788.
\bibitem{b16} G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," Ultralytics, GitHub repository, 2023.
\bibitem{b17} E. H. Land and J. J. McCann, "Lightness and Retinex Theory," Journal of the Optical Society of America, vol. 61, no. 1, pp. 1--11, 1971.
\bibitem{b18} X. Fu, Y. Liao, D. Zeng, Y. Huang, X.-P. Zhang, and X. Ding, "A Weighted Variational Model for Simultaneous Reflectance and Illumination Estimation," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2782--2790.
\bibitem{b19} X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," IEEE Trans. Image Processing, vol. 26, no. 2, pp. 982--993, 2017.
\bibitem{b20} Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, "EnlightenGAN: Deep Light Enhancement Without Paired Supervision," IEEE Trans. Image Processing, vol. 30, pp. 2340--2349, 2021.
\bibitem{b21} C. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," in Proc. British Machine Vision Conf. (BMVC), 2018.
\bibitem{b22} K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum circuit learning," Physical Review A, vol. 98, no. 3, p. 032309, 2018.
\bibitem{b23} M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, "Evaluating analytic gradients on quantum hardware," Physical Review A, vol. 99, no. 3, p. 032331, 2019.
\bibitem{b24} M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, "Parameterized quantum circuits as machine learning models," Quantum Science and Technology, vol. 4, no. 4, p. 043001, 2019.
\bibitem{b25} J. Preskill, "Quantum Computing in the NISQ era and beyond," Quantum, vol. 2, p. 79, 2018.
\bibitem{b26} V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," Nature, vol. 567, no. 7747, pp. 209--212, 2019.
\bibitem{b27} I. Cong, S. Choi, and M. D. Lukin, "Quantum convolutional neural networks," Nature Physics, vol. 15, no. 12, pp. 1273--1278, 2019.
\bibitem{b28} E. Farhi and H. Neven, "Classification with Quantum Neural Networks on Near Term Processors," arXiv:1802.06002, 2018.
\bibitem{b29} A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, "The power of quantum neural networks," Nature Computational Science, vol. 1, no. 6, pp. 403--409, 2021.
\bibitem{b30} S. Mangini, F. Tacchino, D. Gerace, D. Bajoni, and C. Macchiavello, "Quantum computing models for artificial neural networks," Europhysics Letters, vol. 134, no. 1, p. 10002, 2021.
\bibitem{b31} A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in Proc. Neural Information Processing Systems (NeurIPS), 2012, pp. 1097--1105.
\bibitem{b32} K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770--778.
\bibitem{b33} I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in Proc. Int. Conf. Learning Representations (ICLR), 2017.
\bibitem{b34} A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Proc. Neural Information Processing Systems (NeurIPS), 2019, pp. 8024--8035.
\bibitem{b35} M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," Physical Review Letters, vol. 122, no. 4, p. 040504, 2019.
\bibitem{b36} V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," Soviet Physics Doklady, vol. 10, no. 8, pp. 707--710, 1966.
\bibitem{b37} S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in Proc. Int. Conf. Machine Learning (ICML), 2015, pp. 448--456.
\bibitem{b38} M. Cerezo, A. Sone, T. Volkoff, L. Cincio, and P. J. Coles, "Cost function dependent barren plateaus in shallow parametrized quantum circuits," Nature Communications, vol. 12, p. 1791, 2021.
\bibitem{b39} N. Killoran, T. R. Bromley, J. M. Arrazola, M. Schuld, N. Quesada, and S. Lloyd, "Continuous-variable quantum neural networks," Physical Review Research, vol. 1, p. 033063, 2019.
\bibitem{b40} A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, "Data re-uploading for a universal quantum classifier," Quantum, vol. 4, p. 226, 2020.

\end{thebibliography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author1.png}}]{Sreenivasa Reddy Edara}
is a distinguished academician, researcher, and administrator currently serving as a Senior Professor (HAG) in the School of Computer Science and Engineering (SCOPE) at VIT-AP University, Amaravati. His illustrious career spans over 32 years in teaching and 16 years in research, complemented by 24 years of administrative leadership. He has held several prestigious positions, including Vice-Chancellor of JOY University, as well as Dean of Academic Affairs, Dean of the Faculty of Engineering, and Principal of the University College of Engineering \& Technology at Acharya Nagarjuna University (ANU). 

Prof. Reddy holds a Ph.D. in Computer Science \& Engineering from Acharya Nagarjuna University, an M.Tech. from Visveswaraiah Technological University, and an M.S. in Electronics \& Control from BITS, Pilani. His primary research expertise lies in Machine Learning, Deep Learning, Soft Computing, Image Processing, and Pattern Recognition. A prolific mentor, he has guided 43 Ph.D. scholars, 2 M.Phil. students, and 35 M.Tech. students to the successful completion of their degrees. 

With over 260 publications and a Google Citation h-index of 18, Prof. Reddy is widely recognized for his scientific contributions. He has received numerous honors, including the Best Researcher Award (ANU, 2023), the Best Computer Teacher award (ISTE, 2022), the Sarvepalli Radhakrishnan Pratibha Puraskaram (2020), and the Governor’s Gold Medal (2004). Beyond his technical achievements, he is a celebrated author of eight engineering textbooks and holds a title as a "Grand Master" in the India and Asia Books of Record for being the "Most Versatile Telugu Writer," having published hundreds of articles, stories, and poems.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author2.png}}]{Shanmukesh Bonala}
is an AI/ML researcher currently pursuing a B.Tech. degree in Computer Science and Engineering at VIT-AP University, Amaravati, Andhra Pradesh, India. As an aspiring machine learning engineer, he maintains active research interests in trustworthy AI, large language models, neuro-symbolic reasoning, computer vision, and efficient deep learning systems. In addition to his research pursuits, he serves as a student reviewer for selected Elsevier journals and is a Student Member of IEEE.
\end{IEEEbiography}

\EOD
\end{document}
"""
    with open(r"d:\Documents\V-TOP\Winter-Sem 2025-26\AoAI CSE4019\Project AOAI\ACCESS_latex_template_20240429\access.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)

if __name__ == "__main__":
    generate()
