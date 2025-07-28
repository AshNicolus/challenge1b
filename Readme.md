# 📦 Adobe Hackathon Round 1B — Docker Instructions

## 🚀 Build the Docker Image

Make sure you're in the project folder containing the `Dockerfile`, `main.py`, and `model/` directory.

```bash
docker build --platform linux/amd64 -t mysolution:round1b .
docker run --rm --network none -v "${PWD}:/app" mysolution:round1b


