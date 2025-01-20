import torch


if __name__ == '__main__':
    print(torch.backends.mps.is_available())

    tensor0d = torch.tensor(1)

    tensor1d = torch.tensor([1, 2, 3])

    tensor2d = torch.tensor([[1, 2, 3],
                             [3, 4, 5]])

    tensor3d = torch.tensor([[[1, 2], [3, 4]],
                             [[5, 6], [7, 8]]])

    floatvec = torch.tensor([1.0, 2.0, 3.0])

    print(tensor0d)
    print(tensor1d)
    print(tensor2d)
    print(tensor3d)

    print(tensor1d.dtype)
    print(floatvec.dtype)

    print(tensor3d.shape)

    print(tensor2d.view(3, 2))
    print(tensor2d.T)

    print("- matmul and @ operation")
    print(tensor2d.matmul(tensor2d.T))
    print(tensor2d @ tensor2d.T)
