
def fuse_feats(feats_1, feats_2):
    if len(feats_1) != len(feats_2):
        raise ValueError()
    out_feats = []
    for i in range(len(feats_1)):
        out_feats.append(feats_1[i]+feats_2[i])
    return out_feats

def zero_module(module):
    """
    Zero out the parameters of a module.
    """
    for p in module.parameters():
        p.requires_grad = False
        p.zero_()
        p.requires_grad = True
