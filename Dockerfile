FROM amazonlinux:2023

RUN dnf install -y --setopt=install_weak_deps=False \
 python3.12 python3.12-pip libgfortran shadow-utils && \
 useradd -m -u 1000 slam && \
 dnf remove -y shadow-utils && \
 dnf clean all && rm -rf /var/cache/dnf

WORKDIR /usr/src/app

# Copy the pipeline scripts from the submodule (guard AFTER copy)
COPY lib/slam-sigsim/sub_python ./lib/slam-sigsim/sub_python
RUN test -d lib/slam-sigsim/sub_python || (echo "ERROR: submodule missing" >&2 && exit 1)

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
 python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY src src

RUN chown -R slam:slam /usr/src/app
# /data is the intra-container scratch root the plugin writes to; create it
# and hand it to slam, since uid 1000 cannot mkdir at the filesystem root.
RUN mkdir -p /data && chown slam:slam /data
USER slam

ENV OMP_NUM_THREADS=1 \
 OPENBLAS_NUM_THREADS=1 \
 MKL_NUM_THREADS=1 \
 NUMEXPR_NUM_THREADS=1

ENTRYPOINT ["python3.12", "-u"]
CMD ["src/plugin.py"]
