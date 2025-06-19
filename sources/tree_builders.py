import torch as t
from torch import nn

class TreeBuilder:
    
    def build_tree(
        leafs : t.Tensor,
        X : t.Tensor,
        fusion_fn : nn.Module = None
    ):
        raise NotImplementedError
        
class DdTreeBuilder:
    
    def build_tree(
        leafs : t.Tensor,
        X : t.Tensor,
        fusion_fn : nn.Module = None
    ):
        B, L, D, _ = leafs.shape
        assert L % 2 == 0, "Input length must be divisible by 2"
        tree_depth = int(t.log2(t.tensor(L).float()).item())
        hierarchical_memory = [leafs]

        context_summary = [X]  # [B, L, D]
        last_layer = leafs
        last_context = X

        for level in range(tree_depth - 1):
            B, L_prev, D, _ = last_layer.shape
            pairs = last_layer.view(B, L_prev // 2, 2, D, D)
            M1, M2 = pairs[:, :, 0], pairs[:, :, 1]  # [B, L//2, D, D]

            # Summarize the corresponding X embeddings
            X_prev = last_context
            x_pairs = X_prev.view(B, L_prev // 2, 2, D)
            xL, xR = x_pairs[:, :, 0], x_pairs[:, :, 1]  # [B, L//2, D]
            x_shared = (xL + xR) / 2  # shared context

            fused = fusion_fn(
                M1.reshape(-1, D, D),
                M2.reshape(-1, D, D),
                x_shared.reshape(-1, D)
            ).view(B, L_prev // 2, D, D)

            hierarchical_memory.append(fused)
            context_summary.append(x_shared)

            last_layer = fused
            last_context = x_shared

        assert hierarchical_memory[-1].shape[1] == 2
        
        return tree_depth, hierarchical_memory
    
class ReglaTreeBuilder:
    
    def build_tree(
        leafs : t.Tensor,
        X : t.Tensor,
        fusion_fn : nn.Module = None
    ):
        B, L, D, _ = leafs.shape
        assert L % 2 == 0
        tree_depth = int(t.log2(t.tensor(L)).item())
        hierarchical_memory = [leafs]

        last_layer = leafs
        
        for level in range(tree_depth - 1):
            B, L_prev, D, _ = last_layer.shape
            pairs = last_layer.view(B, L_prev // 2, 2, D, D)
            M1, M2 = pairs[:, :, 0], pairs[:, :, 1]

            fused = fusion_fn(M1.reshape(-1, D, D), M2.reshape(-1, D, D)).view(B, L_prev // 2, D, D)
            hierarchical_memory.append(fused)
            last_layer = fused

        assert hierarchical_memory[-1].shape[1] == 2
        
        return tree_depth, hierarchical_memory
    
class LaTreeBuilder:
    
    def build_tree(
        leafs : t.Tensor,
        X : t.Tensor,
        fusion_fn : nn.Module = None
    ):
        B, L, D, _ = leafs.shape

        assert L % 2 == 0, "The sequence len need to be divided by 2 for tree building"

        tree_depth = t.log2(t.tensor(L)).int()
        hierarchical_memory = [leafs]

        last_layer = leafs
        for level in range(tree_depth - 1):

            level_memory = last_layer.reshape(B, L//2**(level + 1), 2, D, D).sum(dim=2)

            hierarchical_memory.append(level_memory)
            last_layer = level_memory

        assert len(hierarchical_memory[-1][0]) == 2, "Invalid head size"
        
        return tree_depth, hierarchical_memory
    
gating_to_builder = {
    "data-driven" : DdTreeBuilder,
    "gated" : ReglaTreeBuilder,
    "summation" : LaTreeBuilder 
}