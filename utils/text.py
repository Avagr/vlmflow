# Ranked by approximate importance
pos_to_class = {
    'ADJ': 24,
    'ADP': 15,
    'ADV': 21,
    'AUX': 11,
    'CCONJ': 12,
    'DET': 16,
    'INTJ': 20,
    'NOUN': 25,
    'NUM': 14,
    'PART': 10,
    'PRON': 17,
    'PROPN': 22,
    'PUNCT': 2,
    'SCONJ': 13,
    'SPACE': 1,
    'SYM': 3,
    'VERB': 23,
    'X': 0
}

class_to_pos = {v: k for k, v in pos_to_class.items()}

replacement_dict = {'<0x0A>': '\n', '<0x0D>': '\r', '<0x09>': '\t'}

colors = ["brown", "red", "yellow", "green", "blue", "purple", "pink", "orange", "black", "white", "gray", "cyan",
          "magenta"]

def retokenize_classes(labels_from: list[tuple[str, str]], tokens_to: list[str]) -> list[list[str]]:
    new_labels = [[] for _ in tokens_to]

    token_to_idx = 0
    pos_in_token_to = 0
    for text, label in labels_from:
        new_labels[token_to_idx].append(label)
        pos_in_token_to += len(text)
        while token_to_idx < len(tokens_to) and pos_in_token_to >= len(tokens_to[token_to_idx]):
            pos_in_token_to -= len(tokens_to[token_to_idx])
            token_to_idx += 1
            if token_to_idx == len(tokens_to) or pos_in_token_to <= 0:
                break
            new_labels[token_to_idx].append(label)

    return new_labels