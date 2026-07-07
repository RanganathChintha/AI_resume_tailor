ATS_LATEX_TEMPLATE = r'''
\documentclass[letterpaper,11pt]{article}

\usepackage[top=0.35in, bottom=0.35in, left=0.55in, right=0.55in]{geometry}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{fancyhdr}

%---------------- Page Style ----------------%
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

%---------------- Section Formatting ----------------%
\titleformat{\section}
  {\vspace{0pt}\scshape\large\bfseries}
  {}{0em}{}[\color{black}\titlerule\vspace{-2pt}]
\titlespacing*{\section}{0pt}{6pt}{4pt}

%---------------- List Formatting ----------------%
\setlist[itemize]{leftmargin=1.5em, topsep=1pt, itemsep=0pt, parsep=0pt}

%---------------- General Formatting ----------------%
\setlength{\parskip}{1pt}
\setlength{\parindent}{0pt}
\raggedright

%---------------- Custom Commands ----------------%
\newcommand{\resumeItem}[1]{\item\small{#1}}

\newcommand{\resumeSubheading}[4]{%
  \textbf{#1} \hfill #2 \\\\
  \textit{\small #3} \hfill \textit{\small #4}
}

\newcommand{\resumeProject}[3]{%
  \textbf{#1} $|$ \textit{\small #2} \hfill \small #3
}

\begin{document}

%==================================================
% HEADER
%==================================================

\begin{center}
    {\Huge \textbf{[Full Name]}} \\[4pt]

    \small
    Phone: [Phone Number] $|$
    Email: [Email Address] \\

    LinkedIn:
    \href{[LinkedIn URL]}
    {[Linkedin/Absolute_Linkedin.com]}
    $|$
    GitHub:
    \href{[GitHub URL]}
    {[github/Absolute_github.com]}
\end{center}

%==================================================
% PROFESSIONAL SUMMARY
%==================================================

\section*{Professional Summary}

[A concise 3-5 line summary highlighting current education/experience, primary technical expertise, domain specialization, biggest achievement, and career objective. Approximately 70-90 words.]

%==================================================
% EDUCATION
%==================================================

\section*{Education}

\resumeSubheading
{[University Name]}
{[City, Country]}
{[Degree Name|GPA/CGPA]}
{[Start Date -- End Date]}

%==================================================
% TECHNICAL SKILLS
%==================================================

\section*{Technical Skills}

\begin{itemize}[leftmargin=1.5em,itemsep=2pt]

\item \textbf{Languages:}
[Comma-separated languages]

\item \textbf{AI / ML:}
[Comma-separated technologies]

\item \textbf{Frameworks:}
[Comma-separated frameworks]

\item \textbf{Backend \& APIs:}
[Comma-separated technologies]

\item \textbf{Databases:}
[Comma-separated databases]

\item \textbf{Tools:}
[Comma-separated tools]

\item \textbf{Core CS:}
[Comma-separated core CS topics]

\end{itemize}

%==================================================
% EXPERIENCE
%==================================================

\section*{Experience}

\resumeSubheading
{[Company Name]}
{[Location]}
{[Role]}
{[Duration]}

\begin{itemize}

\resumeItem{
[A major responsibility with measurable impact, technologies used, and business outcome. Quantify improvements.]
}

\resumeItem{
[Another responsibility with measurable impact.]
}

\resumeItem{
[Another responsibility with measurable impact.]
}

\end{itemize}

%==================================================
% PROJECTS
%==================================================

\section*{Projects}

\resumeProject
{[Project Name]}
{[Tech Stack]}
{[Year]}

\begin{itemize}

\resumeItem{
[Explain what the project does.]
}

\resumeItem{
[Highlight implementation details.]
}

\resumeItem{
[Mention measurable outcomes, optimizations, or impact.]
}

\end{itemize}

\vspace{6pt}

\resumeProject
{[Project Name]}
{[Tech Stack]}
{[Year]}

\begin{itemize}

\resumeItem{
[Explain project objective.]
}

\resumeItem{
[Describe architecture or implementation.]
}

\resumeItem{
[Mention results or business value.]
}

\end{itemize}

%==================================================
% ACHIEVEMENTS & CERTIFICATIONS
%==================================================

\section*{Achievements \& Certifications}

\begin{itemize}

\resumeItem{
[Achievement or Award]
}

\resumeItem{
[Certification or Scholarship]
}

\resumeItem{
[Competition or Recognition]
}

\end{itemize}

\end{document}
'''
