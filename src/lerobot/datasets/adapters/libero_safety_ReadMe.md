# 1) 기존 lerobot env를 그대로 복제 (torch/transformers/lerobot 재설치 불필요)
mamba create -n libero_safety --clone lerobot -y

# 2) 새 env 진입
conda activate libero_safety

# 3) LIBERO-Safety가 필요로 하는 시스템 패키지
apt-get update && apt-get install -y --no-install-recommends \
  unzip libexpat1 libfontconfig1-dev libmagickwand-dev libpython3-stdlib

# 4) 시뮬레이터 전용 파이썬 패키지 (transformers/torch/gym 자리싸움 나는 것들은 제외)
uv pip install --no-cache \
  "hydra-core==1.2.0" "easydict==1.9" \
  "opencv-python==4.6.0.66" "robomimic==0.2.0" "einops==0.4.1" \
  "thop==0.1.1-2209072238" "bddl==1.0.1" "future==0.18.2" \
  "matplotlib==3.5.3" "cloudpickle==2.1.0" "gym==0.25.2" \
  "usd-core>=25.5" "imageio[ffmpeg]" "Wand" "scikit-image"

# 5) fork 클론 + robosuite 포크(1.4.1) --no-deps 설치 + hf-libero 제거
git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git ~/libero-safety
cd ~/libero-safety
git checkout 19ec8df23eedfbb9265bafd3e56495fcebfcfcd0
uv pip install --no-cache --no-deps -e "./third_party/robosuite-1.4"
uv pip uninstall -y hf-libero

# 6) 이 env에서만 fork가 우선되도록 + safety용 config를 별도 경로로 (기존 ~/.libero와 충돌 방지)
conda env config vars set PYTHONPATH="$HOME/libero-safety:${PYTHONPATH}" -n libero_safety
conda env config vars set LIBERO_CONFIG_PATH="$HOME/.libero_safety" -n libero_safety
conda deactivate && conda activate libero_safety   # env var 적용 위해 재진입

uv pip install --upgrade "opencv-python"
uv pip install "mujoco==3.8.1"
uv pip install numba

cd /home #lerobot directory
uv pip install --no-cache -e ".[pi,training]"

# 7) fork가 제대로 잡혔는지 검증 (Dockerfile이 build 타임에 하는 것과 동일)
python -c "
from libero.libero import benchmark
bench = benchmark.get_benchmark_dict()
safety_suites = {'affordance','human_safety','obstacle_avoidance','obstacle_avoidance_human','reasoning_safety'}
missing = safety_suites - bench.keys()
assert not missing, f'still resolving to stock hf-libero: {sorted(bench.keys())}'
print('LIBERO-Safety fork correctly resolved:', sorted(bench.keys()))
"


# 8) 에러방지
앞으로 이 세션에서 새 터미널을 열 때마다, conda activate libero_safety 직후 습관적으로 hash -r 한 번 해주시면 이런 혼선을 피할 수 있습니다. 혹시 또 이상하게 옛날 동작이 나오면 제일 먼저 type -a <명령어>로 PATH 충돌부터 의심해보세요 — 지금 이 컨테이너엔 lerobot 관련 env/venv가 최소 3개(/lerobot/.venv, /home/.venv, miniforge3/envs/lerobot) + 방금 만든 libero_safety까지 있어서 겹칠 여지가 많습니다.