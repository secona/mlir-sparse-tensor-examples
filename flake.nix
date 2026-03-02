{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      llvm = pkgs.llvmPackages_21;

      mkShell = pkgs.mkShell.override { stdenv = llvm.stdenv; };
    in {
      devShells.${system}.default = mkShell {
        packages = [
          llvm.llvm
          llvm.mlir
          llvm.clang-tools
        ];

        hardeningDisable = [ "fortify" ];

        shellHook = ''
          export LLVM_DIR="${llvm.llvm}"
          export MLIR_DIR="${llvm.mlir}"
        '';
      };
    };
}


