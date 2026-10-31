# Start from an official, small Python image matching your local version (3.13).
# "slim" leaves out tools you don't need, so the image is smaller and has less to patch.
FROM python:3.13-slim

# Print logs straight away (not buffered), and don't write .pyc files into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Every command after this runs inside /app in the container.
WORKDIR /app

# Copy ONLY requirements.txt first, then install.
# Docker caches each step: if requirements.txt hasn't changed, the next build
# skips this slow install even when your code has changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the non-root user first, so the copy below can hand it ownership.
RUN useradd --create-home appuser

# --chown makes appuser the owner of every copied file and folder, so
# permissions from whatever machine builds the image no longer matter.
COPY --chown=appuser:appuser . .

USER appuser

# Documents which port the app listens on.
EXPOSE 8000

# What runs when the container starts.
# --host 0.0.0.0: listen for connections from outside the container (127.0.0.1 would only accept connections from inside it)
# --no-browser: there's no browser inside a container
CMD ["python", "-m", "retrieve", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]