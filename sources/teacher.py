from torch import nn
from transformers import AutoConfig, AutoModel

class TinyTransformerTeacher(nn.Module):
    
    def __init__(
        self,
        emb_model   : AutoModel,
        vocab_size  : int = 30522,
    ):
        super().__init__()
        
        self.d_model = emb_model.config.hidden_size
        self.token_emb = emb_model.embeddings
        self.token_emb.requires_grad_(False)
        
        self.attn = nn.MultiheadAttention(self.d_model, num_heads=1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.lm_head = nn.Linear(self.d_model, vocab_size)

    def forward(self, input_ids):
        
        B, L = input_ids.shape
        
        x = self.token_emb(input_ids)
        attn_output, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.ffn(attn_output)
        
        logits = self.lm_head(x)
        return logits, attn_weights