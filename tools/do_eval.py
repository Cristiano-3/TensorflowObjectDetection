import sys, os

sys.path.append('../')

from detectron.utils import voc_eval

imgroot = '../datasets/data/voc_tickets_test/JPEGImages/'
xmlroot = '../datasets/data/voc_tickets_test/Annotations/'
test_imgid_list = [item for item in os.listdir(imgroot)
                        if item.endswith(('.jpg', 'jpeg', '.png', '.tif', '.tiff'))]
test_imgid_list = [item.split('.')[0] for item in test_imgid_list]

voc_eval.do_python_eval(test_imgid_list, test_annotation_path=xmlroot)