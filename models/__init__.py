from .LSTM import LSTM
from .GRU import GRU
from .Attention import Attention

from .GCLSTM import GCLSTM
from .GCGRU import GCGRU

from .STMetaLSTM import STMetaLSTM
from .STMetaGRU import STMetaGRU
from .STMetaAttention import STMetaAttention

from .GWNET import GWNET
from .MTGNN import MTGNN
from .STMetaGCRU import STMetaGCRU
from .STMetaAGCRU import STMetaAGCRU

def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "GRU":
        return GRU
    elif name in ("ATTENTION", "ATTN", "TRANSFORMER"):
        return Attention

    elif name == "GCLSTM":
        return GCLSTM
    elif name == "GCGRU":
        return GCGRU

    elif name == "STMETALSTM":
        return STMetaLSTM
    elif name == "STMETAGRU":
        return STMetaGRU
    elif name == "STMETAGCRU":
        return STMetaGCRU
    elif name in ("STMA", "STMETAAGCRU"):
        return STMetaAGCRU
    elif name in ("STMETAATTN", "STMETAATTENTION", "STMETATRANSFORMER"):
        return STMetaAttention
    elif name in ("GWNET", "GRAPHWAVENET", "GWN"):
        return GWNET
    elif name == "MTGNN":
        return MTGNN


    else:
        raise NotImplementedError
