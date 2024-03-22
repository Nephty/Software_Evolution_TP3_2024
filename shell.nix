let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.matplotlib
      python-pkgs.jupyterlab
      python-pkgs.numpy
      python-pkgs.tqdm
      python-pkgs.scipy
      python-pkgs.seaborn
      python-pkgs.statsmodels
      python-pkgs.git+https://github.com/natarajan-chidambaram/BoDeGHa
    ]))
  ];
}
