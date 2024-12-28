# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from tokenizer.SimpleTokenizerV1 import SimpleTokenizerV1
from tokenizer.SimpleTokenizerV2 import SimpleTokenizerV2
import re

text = "Hello, world. Is this-- a test?"

def preprocess_text(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed


def open_troya_text_file():
    # https://www.gathertales.com/story/the-tale-of-the-trojan-war/sid-822
    with open("troya.txt") as file:
        data = file.read()
    return data




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = open_troya_text_file()
    preprocessed = preprocess_text(data)
    print(len(preprocessed))

    # exmaple to sort vocabs
    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)
    print(vocab_size)

    # create vocabulary
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    ids = tokenizer.encode(text)
    print(ids)
    text_revert = tokenizer.decode(ids)
    print(text_revert)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
