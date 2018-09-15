(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=3.5cm") ("hyperref" "colorlinks")))
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
    "sec:orgc8da660"
    "sec:org01d0b7f"
    "sec:orgab986c4"
    "sec:org81b92f9"
    "sec:org822fc2e"
    "fig:distribution-sectors"
    "fig:gics-level"
    "sec:orgd0f2fb2"
    "sec:orgee61770"
    "sec:orgdb3cd2d"
    "sec:org3a39f6d"
    "sec:org0b799ed"
    "sec:orgeb514e6"
    "sec:org89a789b"
    "fig:keras-nn"
    "sec:orgbd084ae"
    "fig:confusion-matrix"
    "tab:org5b707ed"
    "sec:orgc85a26b"
    "fig:tsne-embedding"
    "fig:silhouette-score"
    "sec:org2ffb474"))
 :latex)

