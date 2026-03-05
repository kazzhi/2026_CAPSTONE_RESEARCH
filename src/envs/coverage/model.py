import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class DroneCarHybridModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Image Branch: Expects (C, H, W) for Torch
        # obs_space.original_space["image"] shape is (2, H, W)
        img_shape = obs_space.original_space["image"].shape 
        
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 2. Vector Branch (Battery, X, Y)
        vec_size = obs_space.original_space["vector"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(vec_size, 32),
            nn.ReLU()
        )

        # 3. Head Calculation
        with torch.no_grad():
            dummy_img = torch.zeros(1, *img_shape)
            cnn_out = self.cnn(dummy_img).shape[1]

        self.logits = nn.Linear(cnn_out + 32, num_outputs)
        self.value = nn.Linear(cnn_out + 32, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        img = obs["image"].float() 
        vec = obs["vector"].float()
        
        img_feats = self.cnn(img)
        vec_feats = self.mlp(vec)
        
        combined = torch.cat([img_feats, vec_feats], dim=-1)
        self._value_out = self.value(combined)
        return self.logits(combined), state

    def value_function(self):
        return self._value_out.flatten()