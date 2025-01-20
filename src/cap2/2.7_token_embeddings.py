# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_ids = torch.tensor([2, 3, 5, 1])

    vocab_size = 6
    output_dim = 3

    # Instantiate an enbedding layer in pytorch
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # Enbedding layer weight matrix
    print(embedding_layer.weight)

    # Get Enbedding vector as example
    print(embedding_layer(torch.tensor([3])))

    # Get Enbedding vector for input ids
    print(embedding_layer(input_ids))

    print("Embedding layers perform a lookup operation")