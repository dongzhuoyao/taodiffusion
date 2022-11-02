
import torch
import numpy as np





# https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792
def slerp_np(val, low, high):  # float, [C], [C]
    omega = np.arccos(
        np.clip(np.dot(low / np.linalg.norm(low),
                high / np.linalg.norm(high)), -1, 1)
    )
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


# https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
def slerp_batch_torch(val, low, high):
    """
    # float,[B,C],[B,C]
    """
    assert len(low.shape) == 2
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res
