# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")

    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

    # Exercie:

    integers = tokenizer.encode("Akwirw ier")
    print(integers)




