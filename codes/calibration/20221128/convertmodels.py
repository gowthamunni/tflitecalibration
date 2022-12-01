import tensorflow as tf
import pathlib

def to_tflite(model, input_shape):
    def representative_dataset():
        data = tf.random.normal(shape = input_shape, dtype = tf.float32)
        yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
                                        
    tflite_models_dir = pathlib.Path("./tflite_models/")
    tflite_models_dir.mkdir(exist_ok = True, parents = True)
    tflite_model_file = tflite_models_dir/"model_{0}_{1}.tflite".format(model.name, model.input_shape)
    tflite_model_file.write_bytes(tflite_model)
