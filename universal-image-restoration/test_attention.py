import torch

import torch.nn as nn

class Fusion_AttentionBlock(nn.Module):
    def __init__(self, low_level_dim, degradation_dim, embed_dim):
        #degradation_dim : 4096, low_level_dim : 512, embed_dim : 512
        super().__init__()
        self.low_level_project = nn.Linear(low_level_dim, embed_dim)
        # self.degradation_project = nn.Linear(degradation_dim, embed_dim)
        self.fusion_project = nn.Linear(embed_dim*2, embed_dim)
        self.embed_dim = embed_dim
        self.init_weights()

    def init_weights(self):
        for module in [self.low_level_project, self.fusion_project]:
            nn.init.constant_(module.bias, 0.)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, low_level_feat, degradation_feat):
        #low_level_feat : [B, 51, 4096], degradation_feat : [B, 512]
        # Project features
        proj_low_level = self.low_level_project(low_level_feat) #shape : [B, 51, 512]
        # proj_degradation = self.degradation_project(degradation_feat.unsqueeze(1)) #shape : [B, 1, 512]
        proj_degradation = degradation_feat.unsqueeze(1) #shape : [B, 1, 512]
        attention_score = torch.einsum('bik,bjk->bij', proj_degradation, proj_low_level) #shape : [B, 1, 51]
        attention = torch.softmax(attention_score, dim=2)
        attended_feat = torch.einsum('bij,bjk->bik', attention, proj_low_level) #shape : [B, 1, 512]
        fused_feat = torch.cat((attended_feat, proj_degradation), dim=2) #shape : [B, 1, 1024]
        fused_feat = self.fusion_project(fused_feat) #shape : [B, 1, embed_dim]
        return fused_feat


# Create random input tensors
low_level_feat = torch.randn(2, 51, 4096)
degradation_feat = torch.randn(2, 512)

# Create an instance of the Fusion_AttentionBlock module
attention_block = Fusion_AttentionBlock(low_level_dim=4096, degradation_dim=512, embed_dim=512)

# Forward pass
attended_features = attention_block(low_level_feat, degradation_feat)

# Print the shape of the flattened features
print(attended_features.shape)