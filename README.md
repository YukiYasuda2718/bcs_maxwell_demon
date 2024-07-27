- [Build an environment](#build-an-environment)
  - [Linux (including WSL)](#linux-including-wsl)
  - [Mac (apple silicon)](#mac-apple-silicon)
- [Plot figures](#plot-figures)

#  Build an environment 

## Linux (including WSL)

1. Make `.env`: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build pytorch_linux`
4. Start a container: `$ docker compose up -d pytorch_linux`
5. Connect to JupyterLab (`http://localhost:9999/lab?`)

## Mac (apple silicon)

- M1 Mac で動作確認済み

1. 環境変数 (`.env`) の作成: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build pytorch_mac`
4. Start a container: `$ docker compose up -d pytorch_mac`
5. Connect to JupyterLab (`http://localhost:7777/lab?`)

# Plot figures

1. Connect to JupyterLab following the above instructions.
2. Perform bootstrap analysis using
3. Plot figures using