#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import csv
import codecs

class YOLO(object):
    def __init__(self):
        self.model_path = './model_data/2cls_smoking_cigarbox.h5'  # model path or trained weights path
        self.anchors_path = './model_data/2cls_smoking_cigarbox_anchors.txt'
        self.classes_path = './model_data/2cls_smoking_cigarbox.txt'
        #优化后的第一版本nice竞品检测模型
        # self.model_path = r'C:\dataset\nice\test\nice_3cls.h5'  # model path or trained weights path
        # self.anchors_path = r'C:\dataset\nice\test\backup\backup\yolo_anchors.txt'
        # self.classes_path = r'C:\dataset\nice\test\backup\backup\voc_classes.txt'
        #nice,smoking 模型
        # self.model_path = r'C:\dataset\nice\nice_smoking\model\nice_smoking.h5'  # model path or trained weights path
        # self.anchors_path = r'C:\dataset\nice\nice_smoking\model\yolo_anchors.txt'
        # self.classes_path = r'C:\dataset\nice\nice_smoking\model\voc_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        # print(" image_data.shape:",image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        font = ImageFont.truetype(font='font/simhei.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        tmp =[]

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]

            # if predicted_class != 'bird'and predicted_class != 'cat' and predicted_class != 'dog' and predicted_class != 'horse' and predicted_class != 'sheep'and predicted_class != 'cow' and predicted_class != 'bear':
            #     continue
            # if predicted_class != 'sheep'or predicted_class != 'cow' or predicted_class != 'bear' :
            #     continue
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            tmp.append(label)
            print(label)
            # print(type(label))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))
            center_y = (top + bottom) / 2
            center_x = (left + right) / 2
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
                draw.point((center_x, center_y), fill=(255, 0, 0))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image, len(out_classes),tmp

    def close_session(self):
        self.sess.close()


def detect_img(yolo):
    traindata_path ='./images'
    save_path='./detection_result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ac_time =time.time()
    file_name = './test/test_%d.csv'%int(ac_time)
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv)
    writer.writerow(['num', "cls", "score"])
    for img in os.listdir(traindata_path):
        try:
            image = Image.open(traindata_path + '/' + img)
            print("单幅图像", image.mode)
        except:
            print('Open Error! Try again!')
        else:
            r_image, num ,tmp= yolo.detect_image(image)
            if len(tmp)>1:
                for i in range(len(tmp)):
                    writer.writerow([img.split('.')[0],tmp[i].split(' ')[0], tmp[i].split(' ')[1]])
                result = np.asarray(r_image)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path + '/' + img, result)
            elif len(tmp) ==1:
                writer.writerow([img.split('.')[0], tmp[0].split(' ')[0], tmp[0].split(' ')[1]])
                result = np.asarray(r_image)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path + '/' + img, result)
            else:
                pass

if __name__ == '__main__':
    time1=time.time()
    detect_img(YOLO())
    time2=time.time()

    print(time2-time1)
