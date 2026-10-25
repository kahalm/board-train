# Trainingsdienst Brett-Erkennung; NVIDIA-PyTorch-Abbild (multi-arch, läuft auf der ARM-GPU der DGX Spark)
FROM nvcr.io/nvidia/pytorch:25.09-py3
WORKDIR /app
COPY server.py net.py final32.pt /app/
EXPOSE 8765
CMD ["python", "/app/server.py"]
