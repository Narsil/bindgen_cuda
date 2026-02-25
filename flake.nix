{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in
        with pkgs;
        {
          default = pkgs.mkShell {
            nativeBuildInputs = [ pkg-config ];
            buildInputs = [
              rustup
              cudaPackages.cudatoolkit
              cudaPackages.cuda_nvcc
            ];
            CUDA_ROOT = "${pkgs.cudaPackages.cudatoolkit}";
            LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          };

        }
      );
    };
}
