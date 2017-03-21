setup <- function() {
  knitr::opts_chunk$set(
    echo = FALSE,
    # Sensible default figure size
    fig.width = 10,
    fig.height = 7,
    # Save all figures to folder
    # fig.path = "../figures/",
    # Generate both PDFs and high-res PNGs
    dev = c("png", "pdf"),
    # dpi = 200,
    # Don't output any unexpected text
    warning = FALSE,
    message = FALSE,
    # Remove '##' from text output
    comment = NA,
    # Cache dir
    cache.path = "../../cache/")

  options(width = 100)

  library(dplyr)
  library(ggplot2)
}
