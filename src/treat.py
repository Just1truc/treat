import torch
from torch                      import nn
from torch.nn                   import functional as F
from einops.layers.torch        import Rearrange


def compute_class_weights(labels, num_classes):
    """
    Args:
        labels (Tensor): shape (batch_size,) - class indices
        num_classes (int): total number of classes
    Returns:
        Tensor: weights of shape (num_classes,)
    """
    class_counts = torch.bincount(labels, minlength=num_classes)
    total = class_counts.sum().item()
    weights = total / (class_counts.float() + 1e-8)  # avoid division by zero
    weights = weights / weights.sum()  # normalize
    return weights


class LocalDualContextGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, d, bias=False)

    def forward(self, M_L, M_R, x_shared):
        """
        M_L, M_R: [B, D, D]
        x_shared: [B, D] — averaged summary of left & right (used for both scores)
        returns: fused summary [B, D, D]
        """
        B, D, _ = M_L.shape
        x_proj = self.gate_proj(x_shared)  # [B, D]

        # Project x into both memory blocks
        score_left = (x_proj.unsqueeze(1) @ M_L @ x_proj.unsqueeze(-1)).reshape(B, 1)
        score_right = (x_proj.unsqueeze(1) @ M_R @ x_proj.unsqueeze(-1)).reshape(B, 1)

        # Normalize scores
        scores = torch.softmax(torch.cat([score_left, score_right], dim=-1), dim=-1)

        # Weighted fusion
        return scores[:, 0].view(B, 1, 1) * M_L + scores[:, 1].view(B, 1, 1) * M_R


class MemoryTree(nn.Module):

    def __init__(
        self,
        leafs : torch.Tensor,
        X : torch.Tensor,
        fusion_fn=None
    ):
        """
        Args:
            leafs (t.Tensor): B, L, D, D
        """
        super().__init__()

        B, L, D, _ = leafs.shape
        assert L % 2 == 0, "Input length must be divisible by 2"
        self.tree_depth = int(torch.log2(torch.tensor(L).float()).item())
        self.hierarchical_memory = [leafs]
        self.fusion_fn = fusion_fn or self.default_fusion_fn

        self.context_summary = [X]  # [B, L, D]
        last_layer = leafs
        last_context = X

        for level in range(self.tree_depth - 1):
            B, L_prev, D, _ = last_layer.shape
            pairs = last_layer.view(B, L_prev // 2, 2, D, D)
            M1, M2 = pairs[:, :, 0], pairs[:, :, 1]  # [B, L//2, D, D]

            # Summarize the corresponding X embeddings
            X_prev = last_context
            x_pairs = X_prev.view(B, L_prev // 2, 2, D)
            xL, xR = x_pairs[:, :, 0], x_pairs[:, :, 1]  # [B, L//2, D]
            x_shared = (xL + xR) / 2  # shared context

            fused = self.fusion_fn(
                M1.reshape(-1, D, D),
                M2.reshape(-1, D, D),
                x_shared.reshape(-1, D)
            ).view(B, L_prev // 2, D, D)

            self.hierarchical_memory.append(fused)
            self.context_summary.append(x_shared)

            last_layer = fused
            last_context = x_shared

        assert self.hierarchical_memory[-1].shape[1] == 2
        self.softmax = nn.Softmax(dim=1)

    def oracle(
        self,
        q : torch.Tensor,
        expected : torch.Tensor = None
    ):
        """
        Args:
            q (t.Tensor): (B, L_k, D)
            expected (t.Tensor): (B, L_k)
        """

        B, L_k, D = q.shape
        get_side_contextual_emb = lambda side_embeddings, query, B: (query.unsqueeze(1) @ side_embeddings @ query.unsqueeze(-1)).reshape(B)

        if expected != None:
            return self._oracle_train(q, expected) # returns loss

        else:

            batch_idx = torch.arange(B, device = q.device)
            all_query_choices = torch.empty(B, L_k, device=q.device)
            for query_id in range(L_k):

                choices = torch.zeros(B, dtype=torch.long, device=q.device)
                for level in range(self.tree_depth - 1, -1, -1):
                    # Only select the memory we need for processing the hard gating. Based on q (B, L_k, D)
                    query = q[:,query_id] # B, D
                    left_embeddings = self.hierarchical_memory[level][batch_idx, choices * 2] # B, These are global choices
                    right_embeddings = self.hierarchical_memory[level][batch_idx, choices * 2 + 1]
                    left_contextual_emb = get_side_contextual_emb(left_embeddings, query, B)
                    right_contextual_emb = get_side_contextual_emb(right_embeddings, query, B)
                    choices = (choices * 2) + (left_contextual_emb < right_contextual_emb)

                all_query_choices[:,query_id] = choices

            return all_query_choices

    def _oracle_train(self, q:torch.Tensor, expected:torch.Tensor):

        B, L_k, D = q.shape
        full_loss = 0.0

        # calculate class weights
        class_weights_per_level = []
        for level in range(self.tree_depth - 1, -1, -1):
            labels = expected // (2 ** level)
            num_classes = self.hierarchical_memory[0].shape[1] // (2 ** level)
            class_weights = compute_class_weights(labels.reshape(labels.shape[0] * labels.shape[1]), num_classes).to(q.device)
            class_weights_per_level.insert(0, class_weights)

        for query_id in range(L_k):
            for level in range(self.tree_depth - 1, -1, -1):
                labels = expected[:, query_id] // (2 ** level) # B, 1 One label per batch for a specific token
                contextual_emb = (q[:, query_id].unsqueeze(1).unsqueeze(2) @ self.hierarchical_memory[level]).squeeze(2) # (B, D) @ (B, L//2**level, D, D) -> B, L//2**level, D
                logits = (contextual_emb @ q[:, query_id].unsqueeze(-1)).squeeze(-1)
                loss = F.cross_entropy(logits.squeeze(-1), labels, weight=class_weights_per_level[level])
                full_loss += loss

        return full_loss


class TreaT(nn.Module):

    def __init__(
        self,
        d_model : int
    ):
        super().__init__()

        self.W_kv = nn.Linear(d_model, 2 * d_model, bias = False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.token_emb = teacher.token_emb
        self.rearrange = Rearrange('b l (k d) -> k b l d', k=2)
        self.local_fusion = LocalDualContextGate(d_model)

    def build_tree(
        self,
        context : torch.Tensor
    ):
        B, L = context.shape
        context_embeddings = self.token_emb(context)
        D = context_embeddings.shape[-1]
        key, values = self.rearrange(self.W_kv(context_embeddings)) # 2, B, L, D
        key = key
        values = values
        leafs = key.unsqueeze(-1) @ values.unsqueeze(-2)
        return MemoryTree(leafs, context_embeddings, fusion_fn=self.local_fusion)

    def forward(
        self,
        query : torch.Tensor,
        memory_tree : MemoryTree,
        expected : torch.Tensor = None
    ):
        """
        Args:
            query (t.Tensor): (B, L_k)
            memory_tree (MemoryTree):
            returns a loss in case of training and prediction in case of eval
        """
        B, L = query.shape
        query_emb = self.token_emb(query)
        query_emb = self.W_q(query_emb)
        return memory_tree.oracle(query_emb, expected)


##### Old Tree Nodes #####

class AttentionFusion(nn.Module):
    '''
    GLA type gating
    '''
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * d * d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, M1, M2):
        """
        M1, M2: [B, D, D]
        returns: fused summary [B, D, D]
        """
        B, D, _ = M1.shape
        concat = torch.cat([M1.view(B, -1), M2.view(B, -1)], dim=-1)
        alpha = torch.sigmoid(self.fc(concat))  # [B, 1]
        return alpha.view(B, 1, 1) * M1 + (1 - alpha.view(B, 1, 1)) * M2


class RefinedAttentionFusion(nn.Module):
    '''
    Refined-GLA type gating
    '''
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
        """
        Args:
            M1, M2: [B, D, D]
        Returns:
            Fused summary: [B, D, D]
        """
        B, D, _ = M1.shape
        concat = torch.cat([M1.reshape(B, -1), M2.reshape(B, -1)], dim=-1)

        g = torch.sigmoid(self.gate(concat))       # Primary gate: [B, 1]
        r = torch.sigmoid(self.refine(concat))     # Refining gate: [B, 1]

        f = (1 - r) * g ** 2 + r * (1 - (1 - g) ** 2)  # ReGLA fusion

        return f.view(B, 1, 1) * M1 + (1 - f.view(B, 1, 1)) * M2
