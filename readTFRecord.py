import tensorflow as tf
import json
from google.protobuf.json_format import MessageToJson

dataset = tf.data.TFRecordDataset("train.tfrecord")
for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d.numpy())
    m = json.loads(MessageToJson(ex))
    print(m['features']['feature'])