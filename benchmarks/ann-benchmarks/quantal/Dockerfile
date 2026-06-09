FROM ann-benchmarks

# Wheels bundle prebuilt libraries for the common embedding dimensions.
RUN pip install quantaldb==0.1.0

# quantal fixes the vector dimension at compile time; dimensions the wheel
# does not bundle are compiled on demand from a source checkout with Zig on
# PATH. Zig is distributed as a single static tarball; no system toolchain
# needed.
ARG ZIG_VERSION=0.16.0
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s /opt/zig-x86_64-linux-${ZIG_VERSION}/zig /usr/local/bin/zig
RUN git clone --branch v0.1.0 --depth 1 https://github.com/PaytonWebber/quantal /opt/quantal
ENV QUANTAL_SRC=/opt/quantal

# Prebuild the benchmark-dataset dimensions the wheel does not cover, so
# runs never pay the compile.
RUN python -c "from quantal import Index; \
    [Index(dim=d) for d in (16, 20, 25, 50, 64, 96, 100, 200)]"

RUN python -c "from quantal import Index; Index(dim=1536)"
