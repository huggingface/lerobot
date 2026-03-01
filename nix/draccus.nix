{
  buildPythonPackage,
  fetchPypi,
  pypaInstallHook,
  setuptoolsBuildHook,
  setuptools,

  mergedeep,
  pyyaml,
  pyyaml-include,
  toml,
  typing-inspect,
}:
buildPythonPackage rec {
  pname = "draccus";
  version = "0.11.5";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-uC4sICcDCuGg8QVRUSX5FOBQwHZqtRjfOgVgoH0Q3ck=";
  };

  nativeBuildInputs = [
    pypaInstallHook
    setuptoolsBuildHook
  ];

  pyproject = true;
  build-system = [ setuptools ];

  dependencies = [
    mergedeep
    pyyaml
    pyyaml-include
    toml
    typing-inspect
  ];

  pythonRelaxDeps = [
    "pyyaml-include"
  ];

  pythonImportsCheck = [ "draccus" ];
}
