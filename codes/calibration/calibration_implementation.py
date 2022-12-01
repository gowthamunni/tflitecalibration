import tensorflow as tf
import numpy as np


def get_range(model_layers):
    layers_convolution = []
    layers_dense = []
    for layer in layers:
        if 'filters'in layer.get_config():
            layers_convolution.append(layer)
        if 'units' in layer.get_config():
            layers_dense.append(layer)
	
    dense_wrange_dict,_ = get_dense_range(layers_dense)
    conv_fil_dict, conv_bias_dict = get_conv_range(layer_convolution)
    
    return dense_wrange_dict

def get_conv_range(layer_dense):

    conv_fil_range = {}
    conv_bias_range = {}
    for layer in layers_convolution:
        filters = layer.filters
        layer_name = layer.name
        filter_weight = layer.get_weights()[0].transpose([3,0,1,2])
        bias = layer.get_weights()[1]


        inter_fil_list = []
        for filter_num in range(filters):
            inter_weights = filter_weight[filter_num, : , :, : ]
            f_min, f_max = np.min(inter_weights), np.max(inter_weights)
            inter_fil_list.append((f_min, f_max))

        b_min, b_max = np.min(bias), np.max(bias)
        
        conv_fil_range[layer_name] = inter_fil_list
        conv_bias_range[layer_name] = (b_min, b_max)
	
    return conv_fil_range, conv_bias_range 


def get_dense_range(layer_dense):
    dense_weight_range = {}
    dense_bias_range = {}
    for layer in layers_dense:
        units = layer.units
        layer_name = layer.name
        dense_weights = layer.get_weights()[0].transpose()
        dense_bias = layer.get_weights()[1]

	d_min, d_max = np.min(dense_weights), np.max(dense_weights)
        
        b_min, b_max = np.min(dense_bias), np.max(dense_bias)

        dense_weight_range[layer_name] = (d_min, d_max)
        dense_bias_range[layer_name] = (b_min, b_max)
        
        return dense_weight_range, dense_bias_range



def int8calibration(range_dict):

    scales = {}

    for layer_name in range_dict:
        scale = (range_dict[layer_name](0) - range_dict[layer_name](1)) / (255)
        scales[layer_name] = scale

    return scales
    
    
if __name__ == "__main__":
    model = tf.keras.applications.VGG16()
    layers = model.layers
    
    dense_dict = get_range(layers)
    calib = int8calibration(dense_dict)
