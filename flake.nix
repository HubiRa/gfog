{
  description = "My app using Hubert's base dev env";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.dotfiles.url = "github:HubiRa/dotfiles";

  outputs = { self, nixpkgs, dotfiles }:
  let
    systems = [ "x86_64-linux" "aarch64-darwin" ];

    forAllSystems = f:
      builtins.listToAttrs (map (system: {
        name = system;
        value = f system;
      }) systems);
  in {
    devShells = forAllSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        default = dotfiles.lib.mkDevShell system
          # extraInputs: project-specific packages
          (with pkgs; [
            pkg-config
            uv
          ])
          # extraShellHook: project-specific env (POSIX!)
          ''
            export PROJECT_NAME="my-app"
            echo "Project dev shell for $PROJECT_NAME"
          '';
      });
  };
}
