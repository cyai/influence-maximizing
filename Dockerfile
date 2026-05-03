FROM ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/cyai/influence-maximizing.git /work

WORKDIR /work

CMD ["bash", "run.sh"]
