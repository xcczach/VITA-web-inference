# VITA 1.5 web inference

Expose an HTTP API for inference.

## Install

```bash
conda create -n <environment-name> python=3.11
conda activate <environment-name>
pip install -r requirements.txt
pip install flash-attn --no-build-isolation # may take a long time; if too long, restart
```

## Usage

Start the server:

```bash
python main.py
```

The API will be available at `http://localhost:9234/vita` by default. You can change the port with `--port` and the API name with `--api-name`.

`test_client.py` provides a sample call to the API.