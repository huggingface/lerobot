{
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,
  poetry-core,

  av,
  datasets,
  deepdiff,
  diffusers,
  draccus,
  einops,
  flask,
  gdown,
  gymnasium,
  h5py,
  huggingface-hub,
  imageio,
  jsonlines,
  numba,
  omegaconf,
  opencv-python-headless,
  pymunk,
  pynput,
  pyzmq,
  rerun-sdk,
  termcolor,
  torch,
  torchvision,
  wandb,
  zarr,

  pyserial,
}:
buildPythonPackage {
  pname = "lerobot";
  version = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2";
  src = fetchFromGitHub {
    owner = "huggingface";
    repo = "lerobot";
    rev = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2";
    hash = "sha256-Kr1UoF9iNGutOIuhYg2FhxtIlbkjoo10ffZJIQNA38Q=";
  };

  nativeCheckInputs = [
    # pytestCheckHook
  ];

  pyproject = true;
  build-system = [
    poetry-core
  ];

  dependencies = [
    av
    datasets
    deepdiff
    diffusers
    draccus
    einops
    flask
    gdown
    gymnasium
    h5py
    huggingface-hub
    imageio
    jsonlines
    numba
    omegaconf
    opencv-python-headless
    pymunk
    pynput
    pyzmq
    rerun-sdk
    termcolor
    torch
    torchvision
    wandb
    zarr
  ];

  pythonRemoveDeps = [
    "cmake"
    "torchcodec" # not packaged
  ];

  pythonRelaxDeps = [
    "av"
    "draccus"
    "gymnasium"
  ];

  checkInputs = [
    pyserial
  ];

  pythonImportsCheck = [ "lerobot" ];
}
