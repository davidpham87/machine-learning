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
    "sec:orgb64b546"
    "sec:orge03247f"
    "sec:orga864785"
    "sec:org1f892c8"
    "sec:org38a1245"
    "fig:distribution-sectors"
    "fig:gics-level"
    "sec:org39dc0fc"
    "sec:orgf060307"
    "sec:org8336ba7"
    "sec:org34e4d70"
    "sec:orgbae949e"
    "sec:org7f8f4d9"
    "sec:orgcc626c2"
    "fig:keras-nn"
    "sec:org52c600f"
    "fig:confusion-matrix"
    "tab:org70e9be4"
    "sec:orgd37413a"
    "fig:tsne-embedding"
    "fig:silhouette-score"
    "sec:org57b47c5"))
 :latex)

