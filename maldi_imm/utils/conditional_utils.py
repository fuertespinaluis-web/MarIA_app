import torch
import torch.nn.functional as F


def get_condition(
    y_species,
    n_classes,
    y_amr=None,
    y_embed_layer_species=None,
    y_embed_layer_amr=None,
    embedding=False,
):
    """
    Get conditioning vector for conditional VAE.
    """
    if embedding:
        y_species_emb = y_embed_layer_species(y_species)

        if y_amr is not None and y_amr.shape[1] > 0:
            all_amr_embeds = y_embed_layer_amr.weight
            batch_embeds = []

            for i in range(y_amr.size(0)):
                y_sample = y_amr[i]
                if torch.any((y_sample > 0) & (y_sample < 1)):
                    weighted_embeds = y_sample.unsqueeze(1) * all_amr_embeds
                    prob_sum = y_sample.sum()
                    if prob_sum > 0:
                        amr_emb = weighted_embeds.sum(dim=0) / prob_sum
                    else:
                        amr_emb = torch.zeros(y_embed_layer_amr.embedding_dim, device=y_sample.device)
                else:
                    idxs = (y_sample == 1).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:
                        amr_emb = torch.zeros(y_embed_layer_amr.embedding_dim, device=y_sample.device)
                    else:
                        amr_emb = y_embed_layer_amr(idxs).mean(dim=0)
                batch_embeds.append(amr_emb)

            y_amr_emb = torch.stack(batch_embeds, dim=0)
            cond = torch.cat([y_species_emb, y_amr_emb], dim=1)
        else:
            cond = y_species_emb
    else:
        y_species_onehot = F.one_hot(y_species, num_classes=n_classes).float()
        if y_amr is not None and y_amr.shape[1] > 0:
            cond = torch.cat([y_species_onehot, y_amr], dim=1)
        else:
            cond = y_species_onehot

    return cond
