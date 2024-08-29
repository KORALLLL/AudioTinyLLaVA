from . import register_vision_tower
from .base import VisionTower
from espnet2.tasks.ssl import SSLTask
import torch

import torch.nn.functional as F

def pad_tensor(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pads the second dimension of a pytorch tensor with zeros to the specified length.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape (bs, init_len, const).
    - target_len (int): The target length to pad to.

    Returns:
    - torch.Tensor: Padded tensor of shape (bs, target_len, const).
    """
    # Get the shape of the input tensor
    bs, init_len, const = tensor.shape
    
    # If the target length is less than or equal to init_len, no padding is needed
    if target_len <= init_len:
        return tensor[:, :target_len, :]

    # Calculate the amount of padding needed
    padding_size = target_len - init_len

    # Pad the tensor (padding is applied to the second dimension)
    # The order of padding is (dim1_left, dim1_right, dim2_left, dim2_right, ...)
    padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), "constant", 0)
    
    return padded_tensor

# # Example usage
# if __name__ == "__main__":
#     # Create an example tensor of shape (2, 3, 4) - (bs=2, init_len=3, const=4)
#     input_tensor = torch.randn(2, 3, 4)
#     print("Original Tensor:")
#     print(input_tensor)
#     print("Shape:", input_tensor.shape)

#     # Pad the tensor to a target length of 5
#     target_len = 5
#     padded_tensor = pad_tensor(input_tensor, target_len)
#     print("\nPadded Tensor:")
#     print(padded_tensor)
#     print("Shape:", padded_tensor.shape)



@register_vision_tower('xeus')
class XEUS(VisionTower): #TODO: flash attention
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _load_model(self, vision_tower_name, **kwargs): # Do we actually need this args? probably yes
        path_to_model = "/home/user27/AudioTinyLLaVA/XEUS/model/xeus_checkpoint.pth" #TODO: maybe put this path to some config
        device = "cuda" #TODO: put it somewhere in config

        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
            None, #TODO: check this arg in docs
            path_to_model,
            device,
        )

        self._vision_tower = xeus_model.half()
        print("\n\n\n\n                        XEUS loaded\n\n\n\n")
    
    def forward(self, x, **kwargs):
        output = self._vision_tower.encode(x, kwargs["wav_lengths"], use_mask=False, use_final_output=False)[0][-1]
        # return pad_tensor(output, 1200)
        return output