import torch
from torch import nn


def test_kmeans():
    from sklearn.cluster import KMeans
    import numpy as np
    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    X = np.random.randn(60000, 1024)
    kmeans = KMeans(n_clusters=40, random_state=0,max_iter=300, verbose=True).fit(X)
    print(kmeans.labels_.shape)
    #kmeans.predict([[0, 0], [12, 3]])
    #print(kmeans.cluster_centers_.shape)


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def test_random_masking_batch():
    x= torch.randn(16, 10, 256)
    mask_ratio = 0.1
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.1)
    print('a')
#test_kmeans()

test_random_masking_batch()






