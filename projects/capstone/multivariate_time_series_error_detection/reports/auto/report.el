(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=3.5cm") ("hyperref" "colorlinks")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
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
    "sec:orgb24a6b5"
    "sec:orgb711bc6"
    "sec:org7a915c3"
    "sec:org7896a9a"
    "sec:orgbb7eeb8"
    "fig:keras-nn"
    "sec:orga558eb8"
    "sec:org6bf7dbd"
    "sec:org06293d3"
    "sec:org90ba10c"
    "sec:org8dd4226"
    "sec:org68e9b17"
    "fig:confusion-matrix"
    "tab:orgab64cd9"
    "sec:org4c7ebc6"
    "fig:tsne-embedding"
    "fig:silhouette-score"
    "sec:orgc4d755f"))
 :latex)

