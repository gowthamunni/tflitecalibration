import tensorflow as tf
import numpy as np
import pathlib

def convert2tflite(model, tensors, path_output):
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_integer-only_quantization
    def representative_data_gen():
        for input_value in tensors:
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    path_output = pathlib.Path(path_output)
    path_output.parent.mkdir(exist_ok=True, parents=True)
    path_output.write_bytes(tflite_model)
    return path_output

class TFliteModel:
    # https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
    '''
    Wrapper class on tflite interpreter
    '''

    def __init__(self, tflite_model_file):
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=str(tflite_model_file))
        self.input_tensor_idxs = []
        for input_tensor_info in self.tflite_interpreter.get_input_details():
            self.input_tensor_idxs.append(input_tensor_info["index"])
        self.output_tensor_idxs = []
        for output_tensor_info in self.tflite_interpreter.get_output_details():
            self.output_tensor_idxs.append(output_tensor_info["index"])
        self.tflite_interpreter.allocate_tensors()

    def __call__(self, input_tensors):
        for input_tensor_idx, input_tensor in zip(self.input_tensor_idxs, input_tensors):
            self.tflite_interpreter.set_tensor(input_tensor_idx, input_tensor)
        self.tflite_interpreter.invoke()
        output_tensors = []
        for output_tensor_idx in self.output_tensor_idxs:
            output_tensor = self.tflite_interpreter.get_tensor(
                output_tensor_idx)
            output_tensors.append(output_tensor)
        return output_tensors

model = TFliteModel("path_to_tflite_model")
output = model([tensors])
