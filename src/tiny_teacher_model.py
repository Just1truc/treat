from torch          import nn
from transformers   import AutoModel

class TinyTransformerTeacher(nn.Module): # model with one single head transformer
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                       d_model=96, vocab_size=30522, max_len=512):
        super().__init__()
        emb_model = AutoModel.from_pretrained(model_name)
        self.d_model = d_model
        self.token_emb = emb_model.embeddings
        self.token_emb.requires_grad_(False)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        attn_output, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.ffn(attn_output)
        logits = self.lm_head(x)
        return logits, attn_weights
