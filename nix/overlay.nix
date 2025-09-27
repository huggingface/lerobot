final: prev: {
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pyFinal: pyPrev: {
      lerobot = pyFinal.callPackage ./lerobot.nix { };
      draccus = pyFinal.callPackage ./draccus.nix { };
    })
  ];
}
