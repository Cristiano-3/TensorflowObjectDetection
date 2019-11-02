# coding: utf-8 
import cv2 
import tensorflow as tf 
import numpy as np
import sys, os
from tqdm import tqdm

sys.path.append('../')
from detectron.models.retinanet import RetinaNet
from detectron.utils.voc_eval import voc_evaluate_detections
from detectron.utils import draw_box_in_img
from configs import cfgs

# build graph, create session, restore
retinanet = RetinaNet('test')

# get image file list
imgroot = '../datasets/data/voc_tickets_test/JPEGImages/'
xmlroot = '../datasets/data/voc_tickets_test/Annotations/'

real_test_imgname_list = [item for item in os.listdir(imgroot)
                         if item.endswith(('.jpg', 'jpeg', '.png', '.tif', '.tiff'))]

# run prediction for each img
all_boxes = []
pbar = tqdm(real_test_imgname_list)

for a_img_name in pbar:
    # read image
    raw_img = cv2.imread(os.path.join(imgroot, a_img_name))#[:, :, ::-1] # BGR2RGB
    raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

    # resize image
    resized_img = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_LINEAR)

    # 
    detected_scores, detected_boxes, detected_categories = retinanet.test_one_batch(np.expand_dims(resized_img, 0))

    # draw & save show
    if True:
        # show_indices = detected_scores >= cfgs.vis_score
        # show_scores = detected_scores[show_indices]
        # show_boxes = detected_boxes[show_indices]
        # show_categories = detected_categories[show_indices]

        final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(resized_img,
                                                                            boxes=detected_boxes,
                                                                            labels=detected_categories,
                                                                            scores=detected_scores,
                                                                            in_graph=False)
        if not os.path.exists(cfgs.test_save_path):
            os.makedirs(cfgs.test_save_path)

        cv2.imwrite(cfgs.test_save_path + '/' + a_img_name.split('.')[0] + '.jpg',
                    final_detections[:, :, ::-1])

    # [y, x, h, w] -> xmin, ymin, xmax, ymax
    ymin = detected_boxes[:, 0] - detected_boxes[:, 2]/2.
    ymax = detected_boxes[:, 0] + detected_boxes[:, 2]/2.
    xmin = detected_boxes[:, 1] - detected_boxes[:, 3]/2.
    xmax = detected_boxes[:, 1] + detected_boxes[:, 3]/2.

    # height & width of resized image. attention!!! shape is HWC not NHWC
    resized_h, resized_w = resized_img.shape[0], resized_img.shape[1]

    # back to origin size
    xmin = xmin * raw_w / resized_w
    xmax = xmax * raw_w / resized_w

    ymin = ymin * raw_h / resized_h
    ymax = ymax * raw_h / resized_h

    # 1xN (stack along axis=0) -> 4xN (transpose) -> Nx4
    boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
    
    # stack as a detection results
    dets = np.hstack(
                        (detected_categories.reshape(-1, 1),
                         detected_scores.reshape(-1, 1),
                         boxes)
                    )

    all_boxes.append(dets)
    pbar.set_description("Eval image %s" % a_img_name)


# # save all detections as .pkl file
# save_dir = cfgs.eval_save_path
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# fw = open(os.path.join(save_dir, 'detections.pkl'), 'w')
# pickle.dump(all_boxes, fw)


# do evaluation
voc_evaluate_detections(all_boxes=all_boxes,
                        test_annotation_path=xmlroot,
                        test_imgid_list=real_test_imgname_list)

