FROM amazonlinux:2023

# Install Python and necessary system libraries
RUN dnf install -y --setopt=install_weak_deps=False \
 python3.12 python3.12-pip libgfortran shadow-utils && \
 useradd -m -u 1000 slam && \
 dnf remove -y shadow-utils && \
 dnf clean all && rm -rf /var/cache/dnf

WORKDIR /usr/src/app

# SLAM-SIGSIM submodule
# Fail early with a clear message if the submodule wasn't initialized
RUN test -d lib/slam-sigsim/sub_python || \
 (echo "ERROR: lib/slam-sigsim is empty. Run: git submodule update --init" >&2 && exit 1)

# Copy the pipeline scripts from the submodule
COPY lib/slam-sigsim/sub_python ./lib/slam-sigsim/sub_python

# Install Python dependencies
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
 python3.12 -m pip install --no-cache-dir -r requirements.txt

# Plugin source
COPY src src

# Set permissions and run as non-root
RUN chown -R slam:slam /usr/src/app
USER slam

# Cap intra-process thread fan-out so memory scales with worker count,
# not num_workers × cpu_count. Important for numba/numpy in containers.
ENV OMP_NUM_THREADS=1 \
 OPENBLAS_NUM_THREADS=1 \
 MKL_NUM_THREADS=1 \
 NUMEXPR_NUM_THREADS=1

ENTRYPOINT ["python3.12", "-u"]
CMD ["src/plugin.py"]