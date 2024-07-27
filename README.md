- [Description](#description)
- [Build environment](#build-environment)
  - [Linux (including WSL)](#linux-including-wsl)
  - [Mac (only for apple silicon)](#mac-only-for-apple-silicon)
- [Plot figures](#plot-figures)

# Description

This repository contains all data and source code used in "Interpretation of the Boundary Current Synchronization as a Maxwell's Demon" by Yuki Yasuda and Tsubasa Kohyama.

#  Build environment 

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