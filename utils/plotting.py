import numpy as np


def aggregate_metric_by_layers(metric, node_layers_dict):
    layers = []
    for layer in range(len(node_layers_dict)):
        layers.append(metric[node_layers_dict[layer]].mean().item())
    return layers

def get_top_tokens(metric, tokenizer, k, layer_from=0, layer_to=None):
    if layer_to is None:
        layer_to = metric.shape[1]
    top_tokens = np.argsort(metric[:, layer_from:layer_to].sum(axis=1))[::-1][:k]
    top_tokens_labels = tokenizer.convert_ids_to_tokens(top_tokens)
    top_tokens_labels = [t.replace('Ġ', "_").replace('Ċ', '\\n') for t in top_tokens_labels]
    return top_tokens, top_tokens_labels
