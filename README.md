- [Description](#description)
- [Build environment](#build-environment)
  - [Linux (including WSL)](#linux-including-wsl)
  - [Mac (only for apple silicon)](#mac-only-for-apple-silicon)
- [Plot figures](#plot-figures)
- [Citation](#citation)

# Description

This repository contains all data and source code used in "A Mechanism of Stochastic Synchronization in the Climate System: An Interpretation of the Boundary Current Synchronization as Maxwell’s Demon" by Yuki Yasuda and Tsubasa Kohyama.

- [Link](https://doi.org/10.1175/JCLI-D-24-0436.1) to our article in Journal of Climate.
- [Link](https://arxiv.org/abs/2408.01133) to our preprint in arXiv.

# Build environment

## Linux (including WSL)

1. Make `.env`: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build pytorch_linux`
4. Start a container: `$ docker compose up -d pytorch_linux`
5. Connect to JupyterLab (`http://localhost:9999/lab?`)

## Mac (only for apple silicon)

1. Make `.env`: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build pytorch_mac`
4. Start a container: `$ docker compose up -d pytorch_mac`
5. Connect to JupyterLab (`http://localhost:7777/lab?`)

# Plot figures

1. Connect to JupyterLab following the above instructions.
2. Perform bootstrap analysis using [bootstrap_analysis.ipynb](./python/notebooks/bootstrap_analysis.ipynb)
3. Plot figures using [plot_figures.ipynb](./python/notebooks/plot_figures.ipynb)

# Citation

```bibtex
@article {
  author = "Yuki Yasuda and Tsubasa Kohyama",
  title = "A Mechanism of Stochastic Synchronization in the Climate System: An Interpretation of the Boundary Current Synchronization as Maxwell’s Demon",
  journal = "Journal of Climate",
  year = "2025",
  publisher = "American Meteorological Society",
  address = "Boston MA, USA",
  volume = "38",
  number = "7",
  doi = "10.1175/JCLI-D-24-0436.1",
  pages= "1573 - 1594",
  url = "https://journals.ametsoc.org/view/journals/clim/38/7/JCLI-D-24-0436.1.xml"
}
```
