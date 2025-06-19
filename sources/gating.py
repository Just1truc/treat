from torch import nn
import torch

class DataDrivenGating(nn.Module):

    def __init__(self, d):
        super().__init__()
        
        self.gate_proj = nn.Linear(d, d, bias=False)

    def forward(self, M_L, M_R, x_shared):

        B, D, _ = M_L.shape
        x_proj = self.gate_proj(x_shared)

        score_left = (x_proj.unsqueeze(1) @ M_L @ x_proj.unsqueeze(-1)).reshape(B, 1)
        score_right = (x_proj.unsqueeze(1) @ M_R @ x_proj.unsqueeze(-1)).reshape(B, 1)

        scores = torch.softmax(torch.cat([score_left, score_right], dim=-1), dim=-1)

        return scores[:, 0].view(B, 1, 1) * M_L + scores[:, 1].view(B, 1, 1) * M_R
    

class RefinedAttentionFusion(nn.Module):
    
    def __init__(self, d):
    
        super().__init__()
        flat_dim = d * d
        self.gate = nn.Sequential(
            nn.Linear(2 * flat_dim, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )
        self.refine = nn.Sequential(
            nn.Linear(2 * flat_dim, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, M1, M2):
        
        B, D, _ = M1.shape
        concat = torch.cat([M1.reshape(B, -1), M2.reshape(B, -1)], dim=-1)

        g = torch.sigmoid(self.gate(concat))
        r = torch.sigmoid(self.refine(concat))

        F = (1 - r) * g ** 2 + r * (1 - (1 - g) ** 2)

        return F.view(B, 1, 1) * M1 + (1 - F.view(B, 1, 1)) * M2
    
gating_to_class = {
    "summation" : lambda _ : None,
    "gated" : RefinedAttentionFusion,
    "data-driven" : DataDrivenGating
}