# REF: based on https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
FROM python:3.11-slim-bookworm

# Install git
RUN apt update
RUN apt install -y git

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4.3 /uv /bin/uv

# Install the project with intermediate layers
# ADD .dockerignore .

# First, install the dependencies
WORKDIR /app
ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Then, install the rest of the project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV RUN_JOBS='True'

CMD ["uv", "run", "runners/segclr_on_2024-08-19.py"]