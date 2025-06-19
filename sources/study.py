# Global
import os
import time
import json
import torch
import torch as t
import torch.nn.functional as F

from tqdm       import tqdm
from matplotlib import pyplot as plt

from transformers       import AutoModel, AutoTokenizer
from transformers       import DataCollatorForLanguageModeling
from sources.teacher    import TinyTransformerTeacher
from torch.utils.data   import DataLoader
from sources.treat      import HierarchicalLinearAttention

from sources.custom_datasets import arg_to_dataset

# Two phases:

# Train teacher transformer
# Save teacher transformer

def train_teacher(
    dataloader  : DataLoader,
    model_name  : str,
    epochs      : int
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = AutoModel.from_pretrained(model_name)
    
    model = TinyTransformerTeacher(emb_model).to(device)
    if os.path.exists("teacher/teacher.pt"):
        print(" === Teacher already trained, skipping training... ===")
        model.load_state_dict(torch.load("teacher/teacher.pt", map_location=device))
        return model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):

        epoch_acc = 0
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, attn_weights = model(input_ids)

            mask = labels != -100
            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / (mask.sum().item() + 1e-7)

            epoch_acc += accuracy

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Accuracy: {epoch_acc / len(dataloader):.4f}")

    torch.save(model.state_dict(), "teacher/teacher.pt")
    return model

def compute_accuracy(predicted_indices, true_indices):
    correct = 0
    total = 0
    variance_loss = 0
    for pred, true in zip(predicted_indices, true_indices):
        # print(pred, true)
        correct += (pred == true).sum().item()
        # print((pred == true))
        variance_loss += t.sqrt((pred - true)**2).mean().item()
        total += len(pred)
        # print(len(pred))
    # print(predicted_indices[0] == true_indices[0])
    return correct / total if total > 0 else 0.0, variance_loss / len(predicted_indices) if total > 0 else 0.0

def train_student(
    teacher : TinyTransformerTeacher,
    dataloader : DataLoader,
    args
):
    teacher.eval()

    # === Initialize HLA ===
    hla = HierarchicalLinearAttention(d_model=teacher.d_model, teacher=teacher, gating=args.gating, format=args.format).cuda()
    hla.train()

    optimizer = torch.optim.Adam(hla.parameters(), lr=1e-4)

    # === Training loop ===
    loss_results = []
    acc_results = []
    var_results = []

    for epoch in range(args.student_epochs):
        for step, batch in enumerate(tqdm(dataloader)):
                
            context_ids = batch["input_ids"].cuda()

            with torch.no_grad():
                logits, attn_weights = teacher(context_ids)
                top1_indices = attn_weights.argmax(dim=-1)
                B, L = top1_indices.shape

            memory_tree = hla.build_tree(context_ids)
            loss = hla.forward(
                query=context_ids,
                memory_tree=memory_tree,
                expected=top1_indices
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # --- Evaluate top-1 match ---
            with torch.no_grad():
                context_embeddings = teacher.token_emb(context_ids)
                predictions = memory_tree.oracle(hla.W_q(context_embeddings), hla.W_v(context_embeddings))
                if args.format == "full":
                    acc, var_loss = compute_accuracy(predictions, top1_indices.unsqueeze(1))
                else:
                    acc = ((predictions.argmax(-1) == (top1_indices // (top1_indices.shape[1] // 2))).sum() / (B * L)).item()

            if step % args.display_freq == 0 and step:
                if args.format == "full":
                    print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f} | Acc: {acc:.4f} | Var: {var_loss:.4f}")
                else:
                    print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f} | Acc: {acc:.4f}")

            acc_results.append(acc)
            loss_results.append(loss.item())
            if args.format == "full":
                var_results.append(var_loss)
            
            if step % args.save_plot_freq == 0 and step:
                if args.format == "full":
                    fig, ax = plt.subplots(1, 3)

                    ax[0].plot(loss_results)
                    ax[0].legend("Loss Curve")
                    ax[1].plot(acc_results, color="green")
                    ax[1].legend("Accuracy Curve")
                    ax[2].plot(var_results)
                    ax[2].legend("Var Curve")

                    plt.savefig(f"plots/auto_save_full_plot_{epoch}_{step}")
                else:
                    fig, ax = plt.subplots(1, 2)

                    ax[0].plot(loss_results)
                    ax[0].legend("Loss Curve")
                    ax[1].plot(acc_results, color="green")
                    ax[1].legend("Accuracy Curve")
                    
                    plt.savefig(f"plots/auto_save_bla_plot_{epoch}_{step}")

def study(args):
    
    dataset = arg_to_dataset[args.dataset](args.emb_model, args.depth)
    
    tokenizer = AutoTokenizer.from_pretrained(args.emb_model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    
    teacher = train_teacher(dataloader, args.emb_model, args.teacher_epochs)
    train_student(teacher, dataloader, args)
    