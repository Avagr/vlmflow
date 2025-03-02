import numpy as np
import numpy_indexed as npi
from tqdm.auto import tqdm


def get_token_layer_matrices(vocab_size, num_layers, dataset_metrics):
    table = dataset_metrics['table']
    node_pos = dataset_metrics['node_pos']
    degrees = dataset_metrics['degrees']
    contributions = dataset_metrics['contributions']
    closeness_pair = dataset_metrics['closeness']
    betweenness = dataset_metrics['betweenness']

    tl_img_centr = np.zeros((vocab_size, num_layers), dtype=float)
    tl_txt_centr = np.zeros((vocab_size, num_layers), dtype=float)
    tl_inter_centr = np.zeros((vocab_size, num_layers), dtype=float)
    tl_closeness_centr = np.zeros((vocab_size, num_layers), dtype=float)
    tl_closeness = np.zeros((vocab_size, num_layers), dtype=float)
    tl_in_degree = np.zeros((vocab_size, num_layers), dtype=float)
    tl_out_degree = np.zeros((vocab_size, num_layers), dtype=float)

    token_occurrence = np.zeros(vocab_size, dtype=float)

    for i in table.idx:
        ids = table.generated_ids[i]
        unique, counts = np.unique(ids[:-1], return_counts=True)
        token_occurrence[unique] += counts
        for j in range(len(betweenness['img_centrality'][i])):
            c_centr = closeness_pair['closeness_centrality'][i][j]
            close = closeness_pair['img_closeness'][i][j]
            img_centrality = betweenness['img_centrality'][i][j]
            txt_centrality = betweenness['txt_centrality'][i][j]
            img_contribs = contributions['img_contrib'][i][j]
            txt_contribs = contributions['txt_contrib'][i][j]
            in_degrees = degrees['in_degrees'][i][j]
            out_degrees = degrees['out_degrees'][i][j]
            layer_num = node_pos[i][j][0]
            token_num = node_pos[i][j][1]
            # Centrality normalization
            img_tokens_mask = (img_contribs == 1) & (layer_num == 0)
            txt_tokens_mask = (txt_contribs == 1) & (layer_num == 0)
            img_norm = (token_num[img_tokens_mask, None] <= token_num[None]).sum(axis=0)
            txt_norm = (token_num[txt_tokens_mask, None] <= token_num[None]).sum(axis=0)
            intersection_centrality = np.float32(np.minimum(txt_centrality, img_centrality))
            inter_norm = np.minimum(img_tokens_mask.sum(), txt_tokens_mask.sum())
            if inter_norm != 0:
                intersection_centrality /= inter_norm
            img_centrality = np.divide(img_centrality, img_norm, out=np.zeros_like(img_centrality, dtype=float),
                                       where=img_norm != 0)
            txt_centrality = np.divide(txt_centrality, txt_norm, out=np.zeros_like(txt_centrality, dtype=float),
                                       where=txt_norm != 0)
            token_ids = ids[token_num]
            index_by_layers = npi.group_by(keys=layer_num)
            for layer, grouped_ids, split_c_centr, split_close, split_img_centr, split_txt_centr, split_inter_centrality, split_in_degrees, split_out_degrees in zip(
                    index_by_layers.unique,
                    index_by_layers.split(token_ids),
                    index_by_layers.split(c_centr),
                    index_by_layers.split(close),
                    index_by_layers.split(img_centrality),
                    index_by_layers.split(txt_centrality),
                    index_by_layers.split(intersection_centrality),
                    index_by_layers.split(in_degrees),
                    index_by_layers.split(out_degrees)
            ):
                index_by_tokens = npi.group_by(keys=grouped_ids)
                cum_c_centrality = np.array([x.sum() for x in index_by_tokens.split(split_c_centr)])
                cum_close = np.array([x.sum() for x in index_by_tokens.split(split_close)])
                cum_img_centrality = np.array([x.sum() for x in index_by_tokens.split(split_img_centr)])
                cum_txt_centrality = np.array([x.sum() for x in index_by_tokens.split(split_txt_centr)])
                cum_inter_centrality = np.array([x.sum() for x in index_by_tokens.split(split_inter_centrality)])
                cum_in_degrees = np.array([x.sum() for x in index_by_tokens.split(split_in_degrees)])
                cum_out_degrees = np.array([x.sum() for x in index_by_tokens.split(split_out_degrees)])
                tl_img_centr[index_by_tokens.unique, layer] += cum_img_centrality
                tl_txt_centr[index_by_tokens.unique, layer] += cum_txt_centrality
                tl_inter_centr[index_by_tokens.unique, layer] += cum_inter_centrality
                tl_closeness[index_by_tokens.unique, layer] += cum_close
                tl_closeness_centr[index_by_tokens.unique, layer] += cum_c_centrality
                tl_in_degree[index_by_tokens.unique, layer] += cum_in_degrees
                tl_out_degree[index_by_tokens.unique, layer] += cum_out_degrees

    return {
        'tl_img_centr': tl_img_centr,
        'tl_txt_centr': tl_txt_centr,
        'tl_inter_centr': tl_inter_centr,
        'tl_closeness_centr': tl_closeness_centr,
        'tl_closeness': tl_closeness,
        'tl_in_degree': tl_in_degree,
        'tl_out_degree': tl_out_degree,
        'token_occurrence': token_occurrence
    }


def get_token_layer_sequence_matrices(vocab_size, num_layers, sequence_length, dataset_metrics):
    table = dataset_metrics['table']
    node_pos = dataset_metrics['node_pos']
    degrees = dataset_metrics['degrees']
    contributions = dataset_metrics['contributions']
    closeness_pair = dataset_metrics['closeness']
    betweenness = dataset_metrics['betweenness']

    tl_img_centr = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_txt_centr = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_inter_centr = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_closeness_centr = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_closeness = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_in_degree = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)
    tl_out_degree = np.zeros((vocab_size, num_layers, sequence_length), dtype=float)

    token_occurrence = np.zeros((vocab_size, sequence_length), dtype=float)

    for i in tqdm(table.idx):
        ids = table.generated_ids[i]
        for j in range(len(betweenness['img_centrality'][i])):
            unique, counts = np.unique(ids[:-sequence_length + j], return_counts=True)
            token_occurrence[unique, j] += counts
            c_centr = closeness_pair['closeness_centrality'][i][j]
            close = closeness_pair['img_closeness'][i][j]
            img_centrality = betweenness['img_centrality'][i][j]
            txt_centrality = betweenness['txt_centrality'][i][j]
            img_contribs = contributions['img_contrib'][i][j]
            txt_contribs = contributions['txt_contrib'][i][j]
            in_degrees = degrees['in_degrees'][i][j]
            out_degrees = degrees['out_degrees'][i][j]
            layer_num = node_pos[i][j][0]
            token_num = node_pos[i][j][1]
            # Centrality normalization
            img_tokens_mask = (img_contribs == 1) & (layer_num == 0)
            txt_tokens_mask = (txt_contribs == 1) & (layer_num == 0)
            img_norm = (token_num[img_tokens_mask, None] <= token_num[None]).sum(axis=0)
            txt_norm = (token_num[txt_tokens_mask, None] <= token_num[None]).sum(axis=0)
            intersection_centrality = np.float32(np.minimum(txt_centrality, img_centrality))
            inter_norm = np.minimum(img_tokens_mask.sum(), txt_tokens_mask.sum())
            if inter_norm != 0:
                intersection_centrality /= inter_norm
            img_centrality = np.divide(img_centrality, img_norm, out=np.zeros_like(img_centrality, dtype=float),
                                       where=img_norm != 0)
            txt_centrality = np.divide(txt_centrality, txt_norm, out=np.zeros_like(txt_centrality, dtype=float),
                                       where=txt_norm != 0)
            token_ids = ids[token_num]
            index_by_layers = npi.group_by(keys=layer_num)
            for (layer, grouped_ids, split_c_centr, split_close, split_img_centr, split_txt_centr,
                 split_inter_centrality, split_in_degrees, split_out_degrees
                 ) in zip(

                index_by_layers.unique,
                index_by_layers.split(token_ids),
                index_by_layers.split(c_centr),
                index_by_layers.split(close),
                index_by_layers.split(img_centrality),
                index_by_layers.split(txt_centrality),
                index_by_layers.split(intersection_centrality),
                index_by_layers.split(in_degrees),
                index_by_layers.split(out_degrees)
            ):
                index_by_tokens = npi.group_by(keys=grouped_ids)
                cum_c_centrality = np.array([x.sum() for x in index_by_tokens.split(split_c_centr)])
                cum_close = np.array([x.sum() for x in index_by_tokens.split(split_close)])
                cum_img_centrality = np.array([x.sum() for x in index_by_tokens.split(split_img_centr)])
                cum_txt_centrality = np.array([x.sum() for x in index_by_tokens.split(split_txt_centr)])
                cum_inter_centrality = np.array([x.sum() for x in index_by_tokens.split(split_inter_centrality)])
                cum_in_degrees = np.array([x.sum() for x in index_by_tokens.split(split_in_degrees)])
                cum_out_degrees = np.array([x.sum() for x in index_by_tokens.split(split_out_degrees)])
                tl_img_centr[index_by_tokens.unique, layer, j] += cum_img_centrality
                tl_txt_centr[index_by_tokens.unique, layer, j] += cum_txt_centrality
                tl_inter_centr[index_by_tokens.unique, layer, j] += cum_inter_centrality
                tl_closeness[index_by_tokens.unique, layer, j] += cum_close
                tl_closeness_centr[index_by_tokens.unique, layer, j] += cum_c_centrality
                tl_in_degree[index_by_tokens.unique, layer, j] += cum_in_degrees
                tl_out_degree[index_by_tokens.unique, layer, j] += cum_out_degrees

    return {
        'tl_img_centr': tl_img_centr,
        'tl_txt_centr': tl_txt_centr,
        'tl_inter_centr': tl_inter_centr,
        'tl_closeness_centr': tl_closeness_centr,
        'tl_closeness': tl_closeness,
        'tl_in_degree': tl_in_degree,
        'tl_out_degree': tl_out_degree,
        'token_occurrence': token_occurrence
    }
