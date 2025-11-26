## On Linux

# Install WSL
# Install mamba

create mamba env (python 3.10)
pip install -r ./tf/requirements.txt

sudo apt update
sudo apt install -y protobuf-compiler

pip uninstall proto-plus
pip uninstall -y protobuf
pip install "protobuf==3.9.1"
pip uninstall -y protobuf
pip install "protobuf==4.21.12"
pip install tensorflow-addons

 ./init.sh

**Create new python env**
python3 -m venv ~/ENV

**Activate env**<br>
source ~/ENV/bin/activate

**cd to repo folder**<br>
cd /mnt/c/0_Repos/lczero-training/

**start training**<br>
python tf/train.py --cfg tf/configs/example_gpu.yaml --output ./tmp/mymodel.txt