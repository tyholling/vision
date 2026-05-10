## Install the Edge TPU runtime
```
curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
unzip edgetpu_runtime_20221024.zip
cd edgetpu_runtime
sudo ./install.sh

ln -s /usr/local/lib/libedgetpu.*.dylib /opt/homebrew/lib/
```
## Install the PyCoral library
```
brew install python@3.9

pip3.9 install --upgrade pip
pip3.9 install numpy==1.26
pip3.9 install pillow
pip3.9 install --extra-index-url https://google-coral.github.io/py-repo/ pycoral
```
## Run a model on the Edge TPU
```
git clone https://github.com/google-coral/pycoral.git
git clone https://github.com/google-coral/test_data.git

./classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels test_data/inat_bird_labels.txt \
--image test_data/parrot.jpg

./classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite \
--labels test_data/inat_insect_labels.txt \
--image test_data/dragonfly.bmp

./classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite \
--labels test_data/inat_plant_labels.txt \
--image test_data/sunflower.bmp

./classify_image.py \
--model test_data/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
--labels test_data/imagenet_labels.txt \
--image test_data/kite_and_cold.jpg
```
