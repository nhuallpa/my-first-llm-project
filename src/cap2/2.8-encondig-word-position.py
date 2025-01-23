# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import tiktoken
from GPTDatasetV1 import GPTDatasetV1
from torch.utils.data import DataLoader
import os




# Press the green button in the gutter to run the script.

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':

    path = os.getcwd()
    print(path)
    with open(path + "/../resources/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)



    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    print("Embedding layers perform a lookup operation")
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)


    #  GPT model’s absolute embedding approach,
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    #  the embedded input examples that can now be processed by the main LLM modules
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)