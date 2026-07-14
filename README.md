# qmirt-gate-10-sim
GATE 10 Simulation for QMIRT project

## Docker CI/CD

GitHub Actions builds the Docker image on pull requests to `main`.
On pushes to `main` and version tags (`v*`), it also publishes the image to:

`ghcr.io/fanghandev/qmirt-gate-10-sim`

## How to use the `sif ` image on OSPool

1. Create `apptainer_cache` directory in your home directory if it does not exist:

```bash
mkdir -p ~/apptainer_cache
```
2. Pull the image from GitHub Container Registry:

```bash
apptainer pull oras://ghcr.io/fanghandev/qmirt-gate-10-sim-sif:v1.0.0
```
3. Move the image to /ospool/ap40/data/username/qmirt-gate-10-sim-sif:

```bash
mv qmirt-gate-10-sim-sif_v1.0.0.sif /ospool/ap40/data/$USER/qmirt-gate-10-sim.sif
```
