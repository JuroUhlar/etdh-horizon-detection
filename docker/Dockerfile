FROM python:3.14-slim-trixie

# Runtime libs that opencv-python-headless needs on Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Nothing else installed here — each attempt brings its own requirements.txt.
# Install before running:
#   docker compose run --rm horizon sh -c "pip install -q -r attempts/<N>/requirements.txt && python attempts/<N>/horizon_detect.py ..."

CMD ["python", "--version"]
