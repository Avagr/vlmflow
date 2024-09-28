# Ranked by approximate importance
import re

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

colors = {"brown", "red", "yellow", "green", "blue", "purple", "pink", "orange", "black", "white", "beige", "gray",
          "cyan", "magenta", "dark", "light", "bright"}


chatgpt_words = {
    'sunset', 'clouds', 'mountain', 'beach', 'car', 'tree', 'flower', 'sky', 'dog', 'cat',
    'building', 'river', 'lake', 'ocean', 'person', 'bird', 'street', 'bridge', 'bicycle',
    'road', 'forest', 'snow', 'sand', 'grass', 'boat', 'fish', 'waves', 'rocks', 'fire',
    'rain', 'hat', 'glasses', 'fence', 'bench', 'umbrella', 'suitcase', 'airplane', 'flag',
    'ball', 'statue', 'skyscraper', 'tower', 'fountain', 'waterfall', 'bus', 'train',
    'motorcycle', 'balloon', 'tent', 'campfire', 'horse', 'sheep', 'cow', 'treehouse',
    'swing', 'tractor', 'windmill', 'crane', 'moon', 'stars', 'cloud', 'traffic', 'carpet',
    'vase', 'mirror', 'chair', 'sofa', 'lamp', 'piano', 'violin', 'drums', 'microphone',
    'balloons', 'cake', 'gift', 'table', 'plate', 'cup', 'bottle', 'pizza', 'burger',
    'fries', 'ice cream', 'cookie', 'sandwich', 'salad', 'fruit', 'vegetable', 'clock',
    'calendar', 'painting', 'sculpture', 'book', 'newspaper', 'magazine', 'window',
    'door', 'curtain', 'keyboard', 'mouse', 'monitor', 'camera', 'phone', 'laptop',
    'tablet', 'guitar', 'headphones', 'microphone', 'speaker', 'television', 'remote',
    'vibrant', 'dim', 'shiny', 'rusty', 'wet', 'dry', 'blurry', 'sharp', 'textured', 'smooth',
    'transparent', 'opaque', 'colorful', 'monochrome', 'glossy', 'matte', 'distorted',
    'symmetrical', 'asymmetrical', 'curved', 'straight', 'tilted', 'crooked', 'dense', 'sparse',
    'faded', 'bright', 'dark', 'reflective', 'shadowy', 'overexposed', 'underexposed', 'contrasted',
    'foggy', 'clear', 'cloudy', 'sunny', 'rainy', 'snowy', 'stormy', 'muddy', 'icy', 'grassy',
    'dusty', 'windy', 'peaceful', 'chaotic', 'crowded', 'isolated', 'vintage', 'modern',
    'industrial', 'rural', 'urban', 'geometric', 'organic', 'fluid', 'rigid', 'ornate',
    'minimalist', 'delicate', 'coarse', 'gritty', 'polished', 'stained', 'frayed', 'wrinkled',
    'tangled', 'stacked', 'layered', 'patterned', 'striped', 'dotted', 'checkered', 'woven',
    'glowing', 'dull', 'vivid', 'blurred', 'focused', 'zoomed', 'zoomed-out', 'framed',
    'cropped', 'foreground', 'background', 'oversized', 'tiny', 'angled', 'abstract',
    'realistic', 'surreal', 'pixelated', 'grayscale', 'saturated', 'desaturated', 'cracked',
    'peeling', 'fractured', 'twisted', 'melting', 'frozen', 'subtle', 'bold', 'futuristic',
    'ancient', 'weathered', 'pristine'
}




def retokenize_labels(labels_from: list[tuple[str, str]], tokens_to: list[str]) -> list[list[str]]:
    new_labels: list[list[str]] = [[] for _ in tokens_to]

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


def label_tokens_by_substring(tokens_to: list[str], text, substrings: set[str]) -> list[str]:
    pattern = '|'.join(re.escape(word) for word in substrings)
    split_result = re.split(f'({pattern})', text)
    initial_labels = []
    for substr in split_result:
        if substr:
            initial_labels.append((substr, "YES" if substr in substrings else "NO"))
    retokenized = retokenize_labels(initial_labels, tokens_to)
    res = []
    for arr in retokenized:
        if "YES" in arr:
            res.append("YES")
        else:
            res.append("NO")
    return res