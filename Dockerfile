FROM scratch
ADD rootfs.tar.gz /
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
