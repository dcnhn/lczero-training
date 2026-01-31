## On Linux

# Install WSL
# Install mamba

create mamba env (python 3.10)
pip install -r ./tf/requirements.txt

sudo apt update
sudo apt install -y protobuf-compiler

pip uninstall proto-plus
pip uninstall -y protobuf
pip install "protobuf==3.20.3"
./init.sh
<!-- pip uninstall -y protobuf
pip install "protobuf==4.21.12" -->
<!-- pip install tensorflow-addons -->

**Create new python env**
python3 -m venv ~/ENV

**Activate env**<br>
source ~/ENV/bin/activate

**cd to repo folder**<br>
cd /mnt/c/0_Repos/lczero-training/

**start training**<br>
python tf/train.py --cfg tf/configs/example_gpu.yaml --output ./tmp/mymodel.txt

python tf/find_corrupted_files.py merged/

tensorboard --logdir leelalogs --port 6031 --host localhost

# Split data

Training start: 11.12.2025 10:20
28.01.2026: S06 adam, board-s05 s05-single-adam