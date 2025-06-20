import torch as t
import torch.nn.functional as F

from torch import nn
from sources.teacher        import TinyTransformerTeacher
from einops.layers.torch    import Rearrange
from sources.gating         import gating_to_class
from sources.tree_builders  import gating_to_builder, TreeBuilder


def compute_class_weights(labels, num_classes):
    
    class_counts = t.bincount(labels, minlength=num_classes)
    total = class_counts.sum().item()
    weights = total / (class_counts.float() + 1e-8)
    weights = weights / weights.sum()
    return weights


class BinaryMemoryTree(nn.Module):

    def __init__(
        self,
        leafs : t.Tensor,
        X : t.Tensor,
        gating : str,
        fusion_fn=None
    ):
        """
        Args:
            leafs (t.Tensor): B, L, D, D
        """
        super().__init__()
        
        self.context = X
        self.gating = gating
        
        tree_depth, hierarchical_memory = gating_to_builder[gating].build_tree(
            leafs,
            X,
            fusion_fn
        )
        
        self.tree_depth = tree_depth
        self.hierarchical_memory = hierarchical_memory

        self.softmax = nn.Softmax(dim=1)

    def oracle(self, q: t.Tensor, v: t.Tensor, expected: t.Tensor = None):
        """
        Args:
            q (Tensor): (B, L_k, D) query for scoring
            v (Tensor): (B, L_k, D) value-projected query vectors
            expected (Tensor): (B, L_k) ground truth leaf index
        """
        B, L_k, _ = q.shape
        output = t.empty(B, L_k, 2, device=q.device)

        for query_id in range(L_k):
            contextual_emb = (q[:, query_id].unsqueeze(1).unsqueeze(2) @ self.hierarchical_memory[self.tree_depth - 1]).squeeze(2)
            logits = (contextual_emb @ v[:, query_id].unsqueeze(-1)).squeeze(-1)
            output[:, query_id] = logits
        
        if expected != None:
            return F.cross_entropy(output.reshape(B * L_k, 2), (expected // (expected.shape[1] // 2)).reshape(B * L_k))
        return output


class MemoryTree(nn.Module):

    def __init__(
        self,
        leafs : t.Tensor,
        X : t.Tensor,
        gating : str,
        fusion_fn=None
    ):
        """
        Args:
            leafs (t.Tensor): B, L, D, D
        """
        super().__init__()
        
        self.context = X
        self.gating = gating
        
        tree_depth, hierarchical_memory = gating_to_builder[gating].build_tree(
            leafs,
            X,
            fusion_fn
        )
        
        self.tree_depth = tree_depth
        self.hierarchical_memory = hierarchical_memory

        self.softmax = nn.Softmax(dim=1)

    def oracle(self, q: t.Tensor, v: t.Tensor, expected: t.Tensor = None):
        """
        Args:
            q (Tensor): (B, L_k, D) query for scoring
            v (Tensor): (B, L_k, D) value-projected query vectors
            expected (Tensor): (B, L_k) ground truth leaf index
        """
        B, L_k, _ = q.shape
        full_loss = 0.0

        if expected is not None:
            # training
            class_weights_per_level = []
            
            for level in range(self.tree_depth - 1, -1, -1):
                labels = expected // (2 ** level)
                num_classes = self.hierarchical_memory[0].shape[1] // (2 ** level)
                class_weights = compute_class_weights(
                    labels.reshape(-1), num_classes
                ).to(q.device)
                class_weights_per_level.insert(0, class_weights)

            for query_id in range(L_k):

                for level in range(self.tree_depth - 1, -1, -1):
                    labels = expected[:, query_id] // (2 ** level)
                    contextual_emb = (q[:, query_id].unsqueeze(1).unsqueeze(2) @ self.hierarchical_memory[level]).squeeze(2)
                    
                    if self.gating == "rvs":
                        logits = (contextual_emb @ (v[:, query_id].unsqueeze(-1))).squeeze(-1) / v.shape[-1]
                    else:
                        logits = (contextual_emb @ (q[:, query_id].unsqueeze(-1))).squeeze(-1)

                    loss = F.cross_entropy(logits, labels, weight=class_weights_per_level[level])# + F.mse_loss(v[:, query_id], self.context[torch.arange(B), expected[:, query_id]])
                    full_loss += loss

            return full_loss

        else:
            # inference
            all_query_choices = t.empty(B, L_k, dtype=t.long, device=q.device)

            for query_id in range(L_k):
                choices = torch.zeros(B, dtype=torch.long, device=q.device)
                for level in range(self.tree_depth - 1, -1, -1):
                    query = q[:,query_id]

                    left_embeddings = self.hierarchical_memory[level][batch_idx, choices * 2]
                    right_embeddings = self.hierarchical_memory[level][batch_idx, choices * 2 + 1]

                    if self.gating == "rvs":
                        left_contextual_emb = (query.unsqueeze(1) @ left_embeddings @ (v[:, query_id].unsqueeze(-1))).reshape(B) / v.shape[-1]
                        right_contextual_emb = (query.unsqueeze(1) @ right_embeddings @ (v[:, query_id].unsqueeze(-1))).reshape(B) / v.shape[-1]
                    else:
                        left_contextual_emb = (query.unsqueeze(1) @ left_embeddings @ (q[:, query_id].unsqueeze(-1))).reshape(B)
                        right_contextual_emb = (query.unsqueeze(1) @ right_embeddings @ (q[:, query_id].unsqueeze(-1))).reshape(B)

                    choices = (choices * 2) + (left_contextual_emb < right_contextual_emb)

                all_query_choices[:, query_id] = choices

            return all_query_choices

format_to_tree = {
    "full": MemoryTree,
    "bla": BinaryMemoryTree
}

class HierarchicalLinearAttention(nn.Module):

    def __init__(
        self,
        d_model : int,
        teacher : TinyTransformerTeacher,
        gating  : str = "sbs",
        format  : str = "full"
    ):
        super().__init__()

        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        self.gating     = gating
        self.token_emb  = teacher.token_emb
        self.rearrange  = Rearrange('b l (k d) -> k b l d', k=2)
        self.local_fusion = gating_to_class[gating](d_model)
        self.memory_tree_class = format_to_tree[format]

    def build_tree(
        self,
        context : t.Tensor
    ):
        context_embeddings = self.token_emb(context)
        
        key     = self.W_k(context_embeddings)
        values  = self.W_v(context_embeddings)

        leafs = key.unsqueeze(-1) @ values.unsqueeze(-2)

        return self.memory_tree_class(leafs, context_embeddings, self.gating, fusion_fn=self.local_fusion)

    def forward(
        self,
        query : t.Tensor,
        memory_tree : MemoryTree,
        expected : t.Tensor = None
    ):
        """
        Args:
            query (t.Tensor): (B, L_k)
            memory_tree (MemoryTree):
            returns a loss in case of training and prediction in case of eval
        """

        context_emb = self.token_emb(query)
        query_emb = self.W_q(context_emb)
        value_emb = self.W_v(context_emb)
        
        return memory_tree.oracle(query_emb, value_emb, expected)