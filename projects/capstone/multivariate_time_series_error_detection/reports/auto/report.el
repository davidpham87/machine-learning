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
    "sec:org49f5cc4"
    "sec:orgbb3d104"
    "sec:org4beb1d4"
    "sec:org2b16581"
    "sec:org271634d"
    "fig:distribution-sectors"
    "fig:gics-level"
    "sec:org68d454b"
    "sec:org789fa4d"
    "sec:org028bbfd"
    "sec:orga6824d6"
    "sec:org529cc81"
    "sec:org4d037f0"
    "sec:orgf5864a6"
    "fig:keras-nn"
    "sec:org7898ae1"
    "fig:confusion-matrix"
    "tab:orgef0f799"
    "sec:orgfb77745"
    "fig:tsne-embedding"
    "fig:silhouette-score"
    "sec:org888f928"))
 :latex)

