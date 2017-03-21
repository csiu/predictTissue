## Generate everything
rmarkdown::render_site(file.path(PROJHOME, "src/rmd"))

## Generate specific pages
rmarkdown::render_site(file.path(PROJHOME, "src/rmd/samples-h3k4me3.Rmd"))

