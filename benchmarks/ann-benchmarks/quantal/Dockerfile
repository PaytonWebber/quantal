FROM ann-benchmarks

# Zig is distributed as a single static tarball — no system package needed.
ARG ZIG_VERSION=0.16.0-dev.2676+4e2cec265
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL "https://ziglang.org/builds/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s /opt/zig-x86_64-linux-${ZIG_VERSION}/zig /usr/local/bin/zig

# quantal source. Pin to a tag/commit in CI; cloned here for portability.
RUN git clone --depth 1 https://github.com/USER/quantal /opt/quantal
WORKDIR /opt/quantal

# Build one shared library per ann-benchmarks-relevant dimension (dim is
# fixed at compile time). The wrapper selects libquantal-<dim>.so.
RUN mkdir -p /home/app/lib && \
    for d in 25 50 100 128 200 300 768 1536; do \
        zig build -Doptimize=ReleaseFast -Dc-dim=$d && \
        cp zig-out/lib/libquantal.so /home/app/lib/libquantal-$d.so; \
    done

# The ctypes wrapper is the algorithm's only runtime Python dependency.
RUN cp python/quantal.py /home/app/quantal.py
ENV QJ_LIB_DIR=/home/app/lib
ENV PYTHONPATH=/home/app:$PYTHONPATH

WORKDIR /home/app
RUN python3 -c "import numpy, quantal"
