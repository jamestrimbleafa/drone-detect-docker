import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("/app/droneInfGraph401092/saved_model/")
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite","wb").write(tflite_model)