# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from importlib.metadata import version
from GPTDatasetV1 import GPTDatasetV1
from torch.utils.data import DataLoader

import tiktoken

print("tiktoken version:", version("tiktoken"))

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("src/resources/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y:      {y}")

    print("In token id format")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    # processing the inputs along the targets
    print("In text format")
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


    # Working with data loader
    print("Working with data loader")

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    print("Working with data loader with batch size of 8")
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4,
        shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)