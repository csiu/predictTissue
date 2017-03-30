## Generate everything
rmarkdown::render_site(file.path(PROJHOME, "src/rmd"))

## Generate specific pages
rmarkdown::render_site(file.path(PROJHOME, "src/rmd/samples-h3k4me3.Rmd"))
rmarkdown::render_site(file.path(PROJHOME, "src/rmd/features.Rmd"))

rmarkdown::render_site(file.path(PROJHOME, "src/rmd/ex_feature-relative-importance.Rmd"))
