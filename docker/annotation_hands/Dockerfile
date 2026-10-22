# Builds native dependencies on CPU. Model weights are pulled with the Job user's token.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel@sha256:53ab3de62f6101d1e42f9be28623ab7a468a24c070d632f211ed576e30b6abd3 AS build
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" MAX_JOBS=1
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ninja-build ffmpeg libglib2.0-0 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --break-system-packages --no-cache-dir uv==0.12.11 \
    && uv python install 3.10.20 \
    && uv venv --python 3.10.20 /opt/droid \
    && uv venv --python 3.10.20 /opt/egohand \
    && python -m venv --system-site-packages /opt/hand-controller
COPY camera-requirements.txt egohand-requirements.txt worker-requirements.txt /opt/hand-worker/
RUN uv pip install --python /opt/droid/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python /opt/droid/bin/python -r /opt/hand-worker/camera-requirements.txt \
    && uv pip install --python /opt/droid/bin/python torch-scatter==2.1.2+pt28cu128 --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
    && uv pip install --python /opt/droid/bin/python git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a --no-deps
RUN uv pip install --python /opt/egohand/bin/python -r /opt/hand-worker/egohand-requirements.txt \
    && /opt/hand-controller/bin/python -m pip install --no-cache-dir -r /opt/hand-worker/worker-requirements.txt scipy==1.15.3
RUN git clone https://github.com/NVlabs/GEM-X.git /opt/GEM-X \
    && git -C /opt/GEM-X checkout --detach 20555befe2fbbe5b2015e1fff7ff214dc18ea9d6 \
    && uv pip install --python /opt/egohand/bin/python --no-deps -e /opt/GEM-X
COPY camera-adapter-source.tar.gz camera-native-source.tar.gz /tmp/
RUN mkdir -p /opt/camera-adapter /opt/hand-camera \
    && tar -xzf /tmp/camera-adapter-source.tar.gz -C /opt/camera-adapter \
    && tar -xzf /tmp/camera-native-source.tar.gz -C /opt/hand-camera
WORKDIR /opt/hand-camera/release-195fps-verified/DROID-W/thirdparty/lietorch
RUN uv pip install --python /opt/droid/bin/python wheel ninja setuptools==79.0.1 \
    && uv pip install --python /opt/droid/bin/python --no-build-isolation --no-deps .
WORKDIR /opt/hand-camera/release-195fps-verified/DROID-W
RUN /opt/droid/bin/python setup.py build_ext --inplace
RUN for source in /opt/hand-camera/release-195fps-verified/DROID-W/*; do \
      name=$(basename "$source"); \
      if [ ! -e "/opt/hand-camera/DROID-on/$name" ]; then ln -s "$source" "/opt/hand-camera/DROID-on/$name"; fi; \
    done \
    && mkdir -p /opt/hand-camera/moge-joint-vio/weights \
    && ln -s /workspace/hand-assets/camera/moge-small.pt /opt/hand-camera/moge-joint-vio/weights/small.pt \
    && ln -s /workspace/hand-assets/camera /opt/hand-camera/release-195fps-verified/DROID-W/pretrained \
    && ln -s /workspace/hand-cache/camera-artifacts /opt/hand-camera/speed-round24/artifacts
RUN rm -rf /opt/hand-camera/release-195fps-verified/DROID-W/build \
    /opt/hand-camera/release-195fps-verified/DROID-W/thirdparty/lietorch/build /tmp/*.tar.gz

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /opt/droid /opt/droid
COPY --from=build /opt/egohand /opt/egohand
COPY --from=build /opt/hand-controller /opt/hand-controller
COPY --from=build /opt/GEM-X /opt/GEM-X
COPY --from=build /opt/camera-adapter /opt/camera-adapter
COPY --from=build /opt/hand-camera /opt/hand-camera
COPY image-version.json health.py /opt/hand-worker/
ENV PATH=/opt/hand-controller/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/user XDG_CACHE_HOME=/home/user/.cache
ENV PYTHONPATH=/opt/camera-adapter/src GDP_TRT_UPDATE=1
ENV DROID_LIBRARY_PATH=/opt/droid/lib/python3.10/site-packages/tensorrt_libs:/opt/droid/lib/python3.10/site-packages/nvidia/cu12/lib:/opt/droid/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64
ENV EGO_LIBRARY_PATH=/opt/egohand/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/egohand/lib/python3.10/site-packages/nvidia/cu13/lib:/opt/egohand/lib/python3.10/site-packages/tensorrt_libs:/usr/local/cuda/lib64
RUN LD_LIBRARY_PATH="$DROID_LIBRARY_PATH" /opt/droid/bin/python -c "import sys; sys.path.insert(0,'/opt/hand-camera/release-195fps-verified/DROID-W'); import torch,lietorch,droid_backends,moge,timm,cv2,tensorrt; from pathlib import Path; from importlib.metadata import distribution; assert torch.__version__=='2.8.0+cu128'; assert tensorrt.__version__=='10.13.3.9'; assert Path(distribution('nvidia-cudss-cu12').locate_file('nvidia/cu12/lib/libcudss.so.0')).is_file()" \
    && LD_LIBRARY_PATH="$EGO_LIBRARY_PATH" /opt/egohand/bin/python -c "import torch,gem,soma,onnxruntime; assert torch.__version__.split('+')[0]=='2.14.1'; assert 'TensorrtExecutionProvider' in onnxruntime.get_available_providers()" \
    && python -c "import torch,numpy,pyarrow,av,scipy; import lerobot_camera_runtime.egocentric.droidw_camera, lerobot_camera_runtime.stages.droidw_slam; assert torch.__version__.split('+')[0]=='2.11.0'"
RUN useradd -m -u 1000 user && mkdir -p /workspace && chown user:user /workspace
USER user
WORKDIR /workspace
EXPOSE 7860
CMD ["python", "/opt/hand-worker/health.py"]
