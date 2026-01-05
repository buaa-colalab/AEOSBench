set -e

project_root=$(dirname $(realpath $0))

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.10

pipenv run pip install /archive/wheels/torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

pipenv run pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021 --no-build-isolation
make install_todd

pipenv run pip install \
    regex \
    "cmake<4.0" \
    "conan<2.0" \
    wheel \
    gymnasium \
    "stable-baselines3[extra]" \
    pymap3d

cd third_party/basilisk
sudo apt install swig
# git checkout c3624e0
# CMAKE_TLS_VERIFY=0 pipenv run python conanfile.py
