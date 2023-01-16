import os
import hashlib
import PIL
import tensorflow as tf
import io
import untangle
from object_detection.utils import dataset_util


dataDir = 'xmls'
recordDir = 'train.tfrecord'

labels = {'cone': 0,
          'cube': 1,
          'foot': 2,
          'hand': 3,
          'face': 4,
          'ball': 5}

writer = tf.io.TFRecordWriter(recordDir)

def xml_to_tf_example(xml_obj):

    full_path = xml_obj.annotation.path.cdata
    filename = xml_obj.annotation.filename.cdata
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(xml_obj.annotation.size.height.cdata)
    height = int(xml_obj.annotation.size.height.cdata)

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    classes = []
    classes_text = []
    truncated = []

    for obj in xml_obj.annotation.object:
        xmin.append(float(obj.bndbox.xmin.cdata) / width)
        ymin.append(float(obj.bndbox.ymin.cdata) / height)
        xmax.append(float(obj.bndbox.xmax.cdata) / width)
        ymax.append(float(obj.bndbox.ymax.cdata) / height)
        classes_text.append(obj.name.cdata.encode('utf8'))
        classes.append(labels[obj.name.cdata])
        truncated.append(int(obj.truncated.cdata))

    # print(height)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
    }))
    return example



for idx, example in enumerate(os.listdir(dataDir)):
    # print(example)
    if example.endswith('.xml'):
        path = os.path.join(dataDir, example)
        xml_obj = untangle.parse(path)
        tf_example = xml_to_tf_example(xml_obj)
        writer.write(tf_example.SerializeToString())
        # print(tf_example.SerializeToString())
writer.close()