#!/usr/bin/env python3.9

import argparse
import numpy
import time

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(*args.model.split('@'))
    interpreter.allocate_tensors()

    input_size = common.input_size(interpreter)
    image = Image.open(args.image).convert('RGB').resize(input_size, Image.LANCZOS)

    input_details = common.input_details(interpreter, 'quantization_parameters')
    scale = input_details['scales']
    zero_point = input_details['zero_points']
    mean = 128
    std = 128

    if abs(scale * std - 1) > 0.001 or abs(mean - zero_point) > 0.001:
        normal_input = (numpy.asarray(image) - mean) / (std * scale) + zero_point
        numpy.clip(normal_input, 0, 255, out=normal_input)
        common.set_input(interpreter, normal_input.astype(numpy.uint8))
    else:
        common.set_input(interpreter, image)

    print()
    for _ in range(5):
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = time.perf_counter() - start
        print('%7.4f ms' % (elapsed * 1000))
    print()

    classes = classify.get_classes(interpreter, 5, 0)
    for c in classes:
        print('%.4f : %s' % (c.score, labels.get(c.id, c.id)))

main()
