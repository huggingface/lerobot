# Camera operators arrive as verified wheels; this image needs no CUDA compiler.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
FROM ${PYTORCH_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libglib2.0-0 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolate the controller, camera and hand dependencies to retain their CUDA ABIs.
FROM base AS build
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN python -m pip install --break-system-packages --no-cache-dir uv==0.12.11 \
    && uv python install 3.10.20 \
    && uv venv --python 3.10.20 /opt/droid \
    && uv venv --python 3.10.20 /opt/egohand \
    && python -m venv --system-site-packages /opt/hand-controller
WORKDIR /opt/hand-worker
COPY camera-requirements.txt egohand-requirements.txt worker-requirements.txt ./
RUN uv pip install --python /opt/droid/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python /opt/droid/bin/python -r /opt/hand-worker/camera-requirements.txt \
    && uv pip install --python /opt/droid/bin/python torch-scatter==2.1.2+pt28cu128 --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
    && uv pip install --python /opt/droid/bin/python git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a --no-deps
RUN uv pip install --python /opt/egohand/bin/python -r /opt/hand-worker/egohand-requirements.txt \
    && /opt/hand-controller/bin/python -m pip install --no-cache-dir -r /opt/hand-worker/worker-requirements.txt
RUN git clone https://github.com/NVlabs/GEM-X.git /opt/GEM-X \
    && git -C /opt/GEM-X checkout --detach 20555befe2fbbe5b2015e1fff7ff214dc18ea9d6 \
    && uv pip install --python /opt/egohand/bin/python --no-deps -e /opt/GEM-X
COPY camera-adapter-source.tar.gz camera-native-source.tar.gz /tmp/
RUN mkdir -p /opt/camera-adapter /opt/hand-camera \
    && tar -xzf /tmp/camera-adapter-source.tar.gz -C /opt/camera-adapter \
    && tar -xzf /tmp/camera-native-source.tar.gz -C /opt/hand-camera
COPY camera-wheels /opt/hand-worker/camera-wheels
COPY verify_camera_wheels.py image-version.json /opt/hand-worker/
RUN python /opt/hand-worker/verify_camera_wheels.py /opt/hand-worker/camera-wheels /opt/hand-worker/image-version.json \
    && uv pip install --python /opt/droid/bin/python --no-deps /opt/hand-worker/camera-wheels/wheels/*.whl
# The frozen camera wrapper expects these source, checkpoint and plan aliases.
RUN for source in /opt/hand-camera/release-195fps-verified/DROID-W/*; do \
      name=$(basename "$source"); \
      if [ ! -e "/opt/hand-camera/DROID-on/$name" ]; then ln -s "$source" "/opt/hand-camera/DROID-on/$name"; fi; \
    done \
    && mkdir -p /opt/hand-camera/moge-joint-vio/weights \
    && ln -s /workspace/hand-assets/camera/moge-small.pt /opt/hand-camera/moge-joint-vio/weights/small.pt \
    && ln -s /workspace/hand-assets/camera /opt/hand-camera/release-195fps-verified/DROID-W/pretrained \
    && ln -s /opt/hand-camera/release-195fps-verified/DROID-W/pretrained /opt/hand-camera/DROID-on/pretrained \
    && ln -s /workspace/hand-cache/camera-artifacts /opt/hand-camera/speed-round24/artifacts
RUN rm -rf /opt/hand-camera/release-195fps-verified/DROID-W/build \
    /opt/hand-camera/release-195fps-verified/DROID-W/thirdparty/lietorch/build /tmp/*.tar.gz

FROM build AS models
COPY model-assets/ /model-assets/
COPY model-sources.json build_model_assets.py /opt/hand-worker/
RUN /opt/egohand/bin/python /opt/hand-worker/build_model_assets.py /model-assets /opt/hand-worker/model-sources.json

FROM base AS runtime
COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /opt/droid /opt/droid
COPY --from=build /opt/egohand /opt/egohand
COPY --from=build /opt/hand-controller /opt/hand-controller
COPY --from=build /opt/GEM-X /opt/GEM-X
COPY --from=build /opt/camera-adapter /opt/camera-adapter
COPY --from=build /opt/hand-camera /opt/hand-camera
COPY image-version.json health.py verify_runtime_imports.py /opt/hand-worker/
COPY camera-startup-source.tar.gz /tmp/
COPY camera-startup-sources.json install_startup_sources.py /opt/hand-worker/
RUN python /opt/hand-worker/install_startup_sources.py /tmp/camera-startup-source.tar.gz /opt/hand-worker/camera-startup-sources.json /opt/hand-worker/image-version.json /opt/hand-camera \
    && rm /tmp/camera-startup-source.tar.gz
ENV PATH=/opt/hand-controller/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/user XDG_CACHE_HOME=/home/user/.cache
ENV PYTHONPATH=/opt/camera-adapter/src GDP_TRT_UPDATE=1 NVIDIA_TF32_OVERRIDE=0
ENV DROID_SITE=/opt/droid/lib/python3.10/site-packages EGO_SITE=/opt/egohand/lib/python3.10/site-packages
ENV DROID_LIBRARY_PATH=${DROID_SITE}/tensorrt_libs:${DROID_SITE}/nvidia/cu12/lib:${DROID_SITE}/nvidia/cudnn/lib:${DROID_SITE}/nvidia/cuda_runtime/lib:${DROID_SITE}/nvidia/cublas/lib:${DROID_SITE}/nvidia/cufft/lib:${DROID_SITE}/nvidia/curand/lib:${DROID_SITE}/nvidia/cuda_nvrtc/lib
ENV EGO_LIBRARY_PATH=${EGO_SITE}/tensorrt_libs:${EGO_SITE}/nvidia/cudnn/lib:${EGO_SITE}/nvidia/cu13/lib:${DROID_SITE}/nvidia/cuda_runtime/lib:${DROID_SITE}/nvidia/cublas/lib:${DROID_SITE}/nvidia/cufft/lib:${DROID_SITE}/nvidia/curand/lib:${DROID_SITE}/nvidia/cuda_nvrtc/lib
RUN if ! getent passwd 1000 >/dev/null; then useradd -m -u 1000 user; fi \
    && mkdir -p /workspace /home/user/.cache && chown -R 1000:1000 /workspace /home/user
USER 1000
WORKDIR /workspace
COPY --from=models --chown=1000:1000 /model-assets /workspace/hand-assets
COPY model-sources.json build_model_assets.py /opt/hand-worker/
RUN python /opt/hand-worker/build_model_assets.py /workspace/hand-assets /opt/hand-worker/model-sources.json --verify-only
RUN python /opt/hand-worker/verify_runtime_imports.py all
EXPOSE 7860
CMD ["python", "/opt/hand-worker/health.py"]
