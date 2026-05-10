```
brew install python@3.9

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
