import torch


def get_torch_device():
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Using CPU!')
    else:
        print('CUDA is available!  Using GPU!')

    return torch.device("cuda:0" if train_on_gpu else "cpu")
