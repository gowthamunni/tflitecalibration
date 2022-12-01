import tensorflow as tf
from convertmodels import to_tflite


MODELS = {"Resnet50":lambda input_shape:tf.keras.applications.resnet50.ResNet50(include_top = True, input_shape = input_shape, weights = None),
        "efficientnet":lambda input_shape:tf.keras.applications.efficientnet.EfficientNetB0(include_top = True, input_shape = input_shape, weights = None)}

input_shape_list = [[1270, 720, 3],[1920, 1080, 3], [3840, 2160, 3]]

def Execute(models_dict = MODELS, input_shape_list = input_shape_list):
    for model_name in MODELS:
        for input_shapes in input_shape_list:
            model = MODELS[model_name](input_shapes)
           
            model.save("./h5models/{0}_{1}.h5".format(model.name, model.input_shape))
            print(model.name, model.input_shape)
            convert_to_tflite(model)


def convert_to_tflite(model):
    input_shape = model.input_shape

    if None in input_shape:
        input_shape = list(input_shape)
        input_shape[0] = 1

    to_tflite(model, input_shape)

if __name__ == "__main__":
    Execute()
