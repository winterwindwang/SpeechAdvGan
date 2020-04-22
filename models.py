import math
import torch
import torchvision
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def prepare_model_setting(label_count, sample_rate, clip_duration_ms,
                          window_size_ms, window_stride_ms,
                          dct_coefficient_count):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)

    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples':desired_samples,
        'window_size_samples':window_size_samples,
        'spectrogram_length':spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count':label_count,
        'sample_rate':sample_rate
    }
def create_model(fingerprint_input, model_setting, model_archittecture,
                 is_training, runtime_settings=None):
    print(fingerprint_input)
    if model_archittecture == 'single_fc':
        pass
    elif model_archittecture == 'conv':
        pass
    elif model_archittecture == 'low_latency_conv':
        pass
    elif model_archittecture == 'low_latency_svdf':
        pass
    else:
        raise Exception('Model_architecture augment "' + model_archittecture + '" not recognized , should be one of '
                                                                               '  "single_fc", "conv","low_latency_conv", "low_latency_svdf" ')
def load_variables_from_checkpoint():
    pass

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def create_single_fc_model(fingerprint_input, model_settings, is_training):
    if is_training:
        pass  # placeholder
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = Parameter(torch.Tensor(torch.randn(fingerprint_input, label_count)))
    bias = Parameter(torch.Tensor(label_count))
    logits = torch.matmul(fingerprint_input, weights) + bias
    if is_training:
        return logits  #, dropout_prob
    else:
        return logits

def create_conv_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = Parameter(torch.Tensor(0.5))
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['input_time_size']
    fingerprint_4d = torch.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

    first_filter_width = 9
    first_filter_height = 20
    first_filter_count = 64
    first_weight = Parameter(torch.Tensor(torch.randn([first_filter_height,first_filter_width, 1, first_filter_count])))
    first_bias = Parameter(torch.Tensor(first_filter_count))
    first_conv = torch.nn.Conv2d(fingerprint_4d, first_weight, [1,1,1,1]) + first_bias
    first_relu = torch.nn.ReLU(first_conv)

    if is_training:
        first_dropout = torch.nn.Dropout2d(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = torch.nn.MaxPool2d(first_dropout, [1,2,2,1], [1,2,2,1])
