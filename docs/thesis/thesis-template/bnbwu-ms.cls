%% ****************************************************
%% **                                          **
%% **     Begum Nusrat Bhutto Women University       **
%% **                                                **
%% **    LaTeX Template for MS/MPhil Thesis  **
%% **                                                **
%% **                 Dr Ram Chand                   **
%% **            ram.chandr@bnbwu.edu.pk             **
%% **			ram.chand2k11@yahoo.com				 **
%% **                                                **
%% ****************************************************

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bnbwu-ms}[2025/02/05 v2.0.0 Begum Nusrat Bhutto Women University, Sukkur -- MS/MPhil Thesis Template]


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper,final]{memoir}
%\LoadClass[twoside]{memoir}

%% **************** Packages (Start) *********************

\RequirePackage[utf8]{inputenc}      % Required for inputting international characters
\RequirePackage[T1]{fontenc}         % Output font encoding for international characters
% Use Palatino (mathpazo) if its T1 font metrics are installed; otherwise
% fall back silently to the always-available Computer Modern. This keeps the
% template portable on minimal TeX installs that lack the URW/Palatino TFMs.
\IfFileExists{pplr8t.tfm}{\RequirePackage{mathpazo}}{}
\RequirePackage[english]{babel}      % Load babel if you're unsure about the default language - mostly to be safe
\RequirePackage{xcolor}              % For named colors
\RequirePackage{eso-pic}                 % required to place huge uni logo at the back - on title page
\RequirePackage{graphicx}            % For pictures (no driver option - auto-detected)
\RequirePackage{amssymb}             % At least, for black squares in bullet list
\RequirePackage[pdfusetitle]{hyperref}             % For hyperreferences
\RequirePackage[authoryear,semicolon,sort]{natbib} % for (Ebejer, 2012; Alba, 2013) kind of references; removed 'square' option after viva examination comments

\RequirePackage{longtable}           % For very long tables
\RequirePackage{pdflscape}			 % For landscape tables (instead of portrait)

\IfFileExists{lato.sty}{%
  \usepackage[defaultsans]{lato}		 % Sans font to use
}{% Fallback chain: Helvetica metrics if present, else Computer Modern Sans
  \IfFileExists{phvr8t.tfm}{\renewcommand{\sfdefault}{phv}}{}%
}
\usepackage[printonlyused,withpage]{acronym}

%% ****************** Packages (End) *********************


%% ************ BNB Definitions (Start) *****************

\definecolor{OxfordBlue}{rgb}{0,0.106,0.329}   % Color
%\definecolor{auburn}{rgb}{0.43, 0.21, 0.1}
\definecolor{BNBRed}{rgb}{0.43, 0.21, 0.1}   % BNB Logo Color

% Setup choosen BNBWU crest/logo
\def\logo{{\includegraphics[width=30mm]{images/logo.png}}}

% The year and term the thesis is submitted 
\def\degreedate#1{\gdef\@degreedate{#1}}
% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}
% The name of your supervisor
\def\supervisor#1{\gdef\@supervisor{#1}}
% The name of your co-supervisor
\def\cosupervisor#1{\gdef\@cosupervisor{#1}}
% The name of your department (e.g. Physics, Computer Science, AI)
\def\department#1{\gdef\@department{#1}}
% The name of your faculty
\def\faculty#1{\gdef\@faculty{#1}}
% Course Code
\def\subjectcode#1{\gdef\@subjectcode{#1}}
% The tagline
\def\tagline#1{\gdef\@tagline{#1}}
% The document type, e.g. a dissertation or a thesis
\def\doctype#1{\gdef\@doctype{#1}}
\def\authorID#1{\gdef\@authorID{#1}}

% -- New fields for MS/MPhil --
\def\enrollment#1{\gdef\@enrollment{#1}}        % Enrollment number
\def\program#1{\gdef\@program{#1}}              % Program name (e.g., MS Physics)
\def\specialization#1{\gdef\@specialization{#1}}% Research specialization
\def\examinerint#1{\gdef\@examinerint{#1}}      % Internal examiner
\def\examinerext#1{\gdef\@examinerext{#1}}      % External examiner
\def\chairperson#1{\gdef\@chairperson{#1}}      % Head of Department
\def\dean#1{\gdef\@dean{#1}}                    % Dean of Faculty

%% ************* BNB Definitions (End) ******************


%% ************ Document Options (Start) *****************

\OnehalfSpacing                                  % One and a half line spacing

\setlength{\headsep}{1.5cm}                      % Add space between the header and text

\nouppercaseheads								 % Don't convert titles to Uppercase
\makepagestyle{bnbpage}

\makeevenhead{bnbpage}{\color{gray}\sffamily\small\leftmark}{}{\color{gray}\sffamily\small\rightmark}
\makeoddhead{bnbpage}{\color{gray}\sffamily\small\leftmark}{}{\color{gray}\sffamily\small\rightmark}
\makeevenfoot{bnbpage}{}{\thepage}{}  			 % bnb pagestyle, put page at bottom
\makeoddfoot{bnbpage}{}{\thepage}{}
\makeheadfootruleprefix{bnbpage}{\color{gray}}{}
\makeheadrule{bnbpage}{\textwidth}{\normalrulethickness}
\makepsmarks{bnbpage}{%
	\createmark{chapter}{left}{shownumber}{\@chapapp\ }{. \ }
	\createmark{section}{right}{shownumber}{}{. \ }
}


\setlrmarginsandblock{3.7cm}{2.5cm}{*}           % Set the page margins (for one and two sided docs) 
\checkandfixthelayout                            % Put layout into effect

\graphicspath{{./images/}}                       % Where to look for images (paths) ...
\DeclareGraphicsExtensions{.pdf,.jpeg,.png,.jpg} % Graphics extensions to load

\chapterstyle{ell} % how to draw the different chapter headings: ell, demo, demo2, lyhne, madsen, veelo

\renewcommand*{\chapnumfont}{\sffamily\HUGE\bfseries\color{BNBRed}}  % Chapter titles should be normal
\renewcommand*{\chaptitlefont}{\sffamily\HUGE\bfseries\color{BNBRed}}
\setsecheadstyle{\sffamily\LARGE\bfseries\color{BNBRed}}% Set \section style
\setsubsecheadstyle{\sffamily\Large\color{BNBRed}}% Set \subsection style
\setsubsubsecheadstyle{\sffamily\large\color{BNBRed}}% Set \subsection style
\setsecnumformat{\csname the#1\endcsname\enskip{\color{gray}\textbar}\enskip}

\newsubfloat{figure} % declares a new subfloat element which allows to use \subbottom commands

\renewcommand{\labelitemi}{\color{BNBRed}\scriptsize$\blacksquare$}

\addto{\captionsenglish}{\renewcommand{\bibname}{References}} % because we are using babel we need this
                                                              % instead of just redefining bibname
\setlength{\bibitemsep}{\onelineskip}

\newcommand{\ra}[1]{\renewcommand{\arraystretch}{#1}} % to set array stretch within tables

\hypersetup{%
    colorlinks=false,
%    linkcolor=BNBRed,
%    citecolor=BNBRed,
%    urlcolor=BNBRed,
%    filecolor=magenta, 
    pdfborder={0 0 0},    
}

\feetbelowfloat % we want the footnotes below floats, not wierdly above

\setsecnumdepth{subsubsection}  % three level depth - chapter, section, subsection, subsubsection
\settocdepth{subsection}

\renewcommand*{\cftappendixname}{Appendix\space}
\newcommand{\removelinebreaks}[1]{%
	\begingroup\def\\{ }#1\endgroup}

%% ************* Document Options (End) ******************


%% *************** Environments (Start) ******************

%% *** Title Page (Start) ***
% background image
% taken from http://tug.org/pracjourn/2008-1/mori/mori.pdf
\newcommand\AlCentroPagina[1]{%
\AddToShipoutPicture*{\AtPageCenter{%
\makebox(0,-50){\includegraphics[width=0.5\paperwidth]{#1}}}}}

% The front page
\renewcommand{\maketitle}
{\begingroup
\AlCentroPagina{logowater1}
\parbox[b][0.95\textheight][t]{0.2\textwidth}{\raggedleft\logo} %% this 0.95 is important, otherwise page overflows in next page
\hspace*{2ex}
\textcolor{BNBRed}{\rule{2pt}{0.95\textheight}}
\hspace*{2ex}
\parbox[b][0.95\textheight][t]{0.8\textwidth}{
    \setlength{\parindent}{0pt}
    %%\fontfamily{pag}\selectfont
    \sffamily
    {\Huge\bfseries{\begin{Spacing}{1.15}\textcolor{BNBRed}{\@title}\end{Spacing}}}
    \vspace*{2ex}
    {\large\textit{\@tagline}}\\[1in]
    
    {\LARGE\bfseries \textsc{\@author}} \\[0.5cm]
    \@ifundefined{@enrollment}{}{{\large Enrollment No: \@enrollment}\\[0.5cm]}
    \@ifundefined{@program}{}{{\large \@program}\\[0.3cm]}
    \@ifundefined{@specialization}{}{{\large Specialization: \@specialization}\\[0.5cm]}
    {\large Supervised by \@supervisor} 
    \@ifundefined{@cosupervisor}{\\[2ex]}{\\[2ex]\large Co-supervised by \@cosupervisor}\\[1cm]
    {\large \@department} \\[1ex]
    {\large \@faculty} \\[1ex]
    {\large Begum Nusrat Bhutto Women University, Sukkur} \\
    \vfill
    {\bfseries \@degreedate}\\[\baselineskip]
	\parbox[t]{0.80\linewidth}{\footnotesize\textit{A \@doctype\ submitted in partial fulfilment of the requirements for the degree of \@degree}.}    
}
\thispagestyle{empty}
\if@openright\cleardoublepage\else\clearpage\fi
\endgroup}
%% *** Title Page (End) ***

\newenvironment{certificate}
{\clearpage\thispagestyle{empty}
	{\noindent\Large\bfseries\sffamily\textcolor{BNBRed}{CERTIFICATE}}\\[3mm]
	\setlength{\parindent}{0em}
}
{\if@openright\cleardoublepage\else\clearpage\fi}


\newenvironment{acknowledgements}
{\renewcommand{\abstractname}{\Large\bfseries\sffamily\textcolor{BNBRed}{Acknowledgements}}\abstract}
{\endabstract\if@openright\cleardoublepage\else\clearpage\fi}
       
\addto{\captionsenglish}{\renewcommand{\abstractname}{\Large\bfseries\sffamily\textcolor{BNBRed}{Abstract}}}


\newenvironment{dedication}
  {\clearpage           % we want a new page
   \thispagestyle{empty}% no header and footer
   \vspace*{\stretch{1}}% some space at the top 
   \itshape             % the text is in italics
   \raggedleft          % flush to the right margin
   \textcolor{BNBRed}
  }
  {\par % end the paragraph
   \vspace{\stretch{3}} % space at bottom is three times that at the top
   \if@openright\cleardoublepage\else\clearpage\fi
  }       
       

%%\renewenvironment{abstract}
%%{\begin{alwayssingle} \thispagestyle{empty}
%%\begin{center}
%%\vspace*{0.5cm}
%%{\large \bfseries \textcolor{BNBRed}{Abstract}}
%%\end{center}
%%\singlespacing\enlargethispage{\baselineskip}}
%%{\end{alwayssingle}}

%%{\cleardoublepage\newgeometry{right=1.3cm,top=2.4cm,bottom=2cm}\pagestyle{empty}\begin{center}\bfseries\large\textcolor{OxfordBlue}\abstractname\end{center}\normalsize\onehalfspacing\begin{quotation}}{\end{quotation}\restoregeometry\cleardoublepage}


% STATEMENT OF ORIGINALITY
%
% The originality environment puts a large, bold, centered 
% "Statement of originality" label at the top of the page. The statement 
% of originality itself appears in a quote environment, i.e. tabbed in at 
% both sides, and on its own page.

%\newenvironment{originality}
%{\clearpage
%\begin{center}
%\vspace*{1.5cm}
%{\Large\bfseries\sffamily\textcolor{BNBRed}{Statement of Originality}\par}
%\vspace*{1.5cm}
%{I, the undersigned, declare that this is my own work unless where otherwise acknowledged and referenced. \par}
%\vspace*{1.5cm}
%\renewcommand\arraystretch{1.5}
%\begin{tabular}{ll}
%      {\bfseries Candidate} & \@author \\[15mm]
%      {\bfseries Signed} & \makebox[1.5in]{\hrulefill} \\
%      {\bfseries Date} & \today \\
%\end{tabular}
%\end{center}
%\if@openright\cleardoublepage\else\clearpage\fi}

%% This was updated to the official Statement of Originality (Authentication form)

\newenvironment{originality}
{\clearpage\thispagestyle{empty}
%\begin{figure}[h]
%	\includegraphics[width=2.5in]{images/Loginame.png}
%\end{figure}
%\vspace{3mm}
{\noindent\Large\bfseries\sffamily\textcolor{BNBRed}{DECLARATION BY THE CANDIDATE}}\\

{\setlength{\parindent}{0em}

\noindent I, \textbf{\@author\ (\@enrollment)}, hereby declare that the work presented in this thesis entitled \textbf{``\removelinebreaks{\@title}''} is the result of the work performed by me in the \textbf{\@department, Begum Nusrat Bhutto Women University, Sukkur}, under the supervision of \textbf{\@supervisor} and that it has not been submitted elsewhere for any degree or diploma. I further declare that:

\begin{enumerate}
    \item This \@doctype\ has not been submitted, either in whole or in part, for a degree, diploma, or qualification at this university or any other institution.
    \item The intellectual content of this \@doctype\ is the product of my own work, except where due acknowledgement has been made in the text to all other material used.
    \item All ethical guidelines prescribed by the university for research have been followed in this study.
\end{enumerate}

\noindent In keeping with the general practice, due acknowledgements are made wherever the work is based on findings of other investigations.}\\[5mm]
%%\newcolumntype{L}{>{\arraybackslash}m{2.5in}}
\bgroup
\def\arraystretch{1.2}%
% Second column is a wrapping paragraph column so that long titles /
% multi-author candidate lists do not overflow the page width.
\begin{tabular}{l p{0.62\textwidth}}
{\bfseries Department} & \@department \\
{\bfseries Program} & \@ifundefined{@program}{\@degree}{\@program} \\
{\bfseries Degree} & \@degree \\
{\bfseries Title}  & \removelinebreaks{\@title} \\
{\bfseries Candidate \@ifundefined{@authorID}{}{(ID)}} & \@author\ \@ifundefined{@authorID}{}{(\@enrollment)} \\[0.7in]
{\bfseries Signature of Candidate} & \makebox[2in]{\hrulefill} \\
{\bfseries Date} & \today \\
\end{tabular}\\[1cm]
\egroup
%\noindent 08.02.2018
\if@openright\cleardoublepage\else\clearpage\fi}


% COPYRIGHT
%

\newenvironment{copyrightenv}
{\clearpage to
	~\vfill
	\thispagestyle{empty}
	
	\begin{figure}[h]
		\includegraphics[width=1.5in]{images/logo.png}
	\end{figure}
	
	\noindent Copyright \textcopyright\the\year\ Begum Nusrat Bhutto Women University, Sukkur \\ % Copyright notice
	
	\noindent \textsc{www.bnbwu.edu.pk}\\ % URL
	
	\noindent \textit{First edition, \today} % Printing/edition date
	
	\if@openright\cleardoublepage\else\clearpage\fi}

%% **************** Environments (End) *******************