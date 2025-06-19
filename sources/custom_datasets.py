def load_wikipedia_dataset(emb_model : str, depth : int):
    
    import torch
    import requests

    from torch.utils.data import Dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(emb_model)

    def download_wikitext2():
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
        text = requests.get(url).text
        return [line.strip() for line in text.split('\n') if line.strip()]

    # ==== Dataset Class ====
    class WikiTextDataset(Dataset):
        def __init__(self, lines, tokenizer, block_size=64):
            self.examples = []
            self.tokenizer = tokenizer
            length = len(lines)
            pourcentage = 0.3

            for line in lines[:int(length * pourcentage)]:
                tokens = tokenizer.encode(line, add_special_tokens=True)
                for i in range(0, len(tokens), block_size):
                    block = tokens[i:i+block_size]
                    if len(block) == block_size:
                        self.examples.append(torch.tensor(block))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]


    lines = download_wikitext2()
    lm_dataset = WikiTextDataset(lines, tokenizer, block_size=2**depth)
    
    return lm_dataset

arg_to_dataset = {
    "wikitext2" : load_wikipedia_dataset
}