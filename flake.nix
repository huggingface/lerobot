{
  inputs.nixpkgs.url = "github:nixos/nixpkgs";

  outputs =
    { self, nixpkgs, ... }:
    {
      overlays.default = import ./nix/overlay.nix;
      legacyPackages.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.extend (self.overlays.default);
      devShells.x86_64-linux.default =
        let
          pkgs = self.legacyPackages.x86_64-linux;
        in
        pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (pp: [
              pp.lerobot
            ]))
          ];
        };
    };
}
