import torch
import numpy as np 


def feat_interrupt(feat_batch, _method):
    if _method == 'mix':
        batch_size = len(feat_batch)
        index = torch.randperm(batch_size).to(feat_batch.device)
        feat_pair = feat_batch[index]
        

    elif _method =='mixot':
        import ot 
        batch_size = len(feat_batch)

        #index = torch.randperm(batch_size).to(feat.device)
        M = ot.dist(feat_batch, feat_batch, metric='euclidean').cpu().numpy()
        for _ in range(len(M)):
            M[_][_] = 1e20
        a, b = np.ones((batch_size,)) / batch_size, np.ones((batch_size,)) / batch_size  # uniform distribution on samples
        _ot_result = ot.emd(a, b, M)
        
        index = _ot_result.argmax(1) 
        feat_pair = feat_batch[index]

    noise_ratio = torch.rand((batch_size, 1))
    feat_interrupted = feat_batch*(1-noise_ratio) + feat_pair*noise_ratio

    ret =  torch.cat([feat_interrupted, noise_ratio],1)
    return ret


if __name__=='__main__':
    feat = torch.randn((128, 512))
    _method = 'mixot'
    feat_interrupt(feat, _method)
   