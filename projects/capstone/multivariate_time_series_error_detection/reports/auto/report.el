(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=3.5cm")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "lmodern"
    "geometry"
    "pdflscape")
   (LaTeX-add-labels
    "sec:org59cffe4"
    "sec:org46a4ecf"
    "sec:orgda57672"
    "sec:orge8906e6"
    "sec:org89bc8d0"
    "fig:distribution-sectors"
    "fig:gics-level"
    "sec:org90358cc"
    "sec:org9548d7c"
    "sec:orgdf4002a"
    "sec:orge16cd46"
    "sec:org6549843"
    "sec:org73b2063"
    "sec:orgf675192"
    "fig:keras-nn"
    "sec:orgd2caaf1"
    "fig:confusion-matrix"
    "tab:org9c87c36"
    "tab:org57ac37e"
    "sec:org45ebadd"
    "sec:org6a416c4"
    "fig:tine-embedding"
    "fig:silhouette-score"
    "sec:org8bd0427"))
 :latex)

