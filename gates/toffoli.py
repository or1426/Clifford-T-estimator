from .base import Gate, TGate, CompositeGate
from .cliffords import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate

def CCZGate(a,b,c):
    return CompositeGate([TGate(a), CXGate(b,a), 
                          HGate(b), SGate(b), SGate(b), HGate(b),
                          TGate(b),
                          HGate(b), SGate(b), SGate(b), HGate(b),
                          CXGate(b,a), TGate(b), CXGate(c,b),
                          HGate(c), SGate(c), SGate(c), HGate(c),
                          TGate(c),
                          HGate(c), SGate(c), SGate(c), HGate(c),
                          CXGate(c,a), TGate(c), CXGate(c,b),
                          HGate(c), SGate(c), SGate(c), HGate(c),
                          TGate(c),
                          HGate(c), SGate(c), SGate(c), HGate(c),
                          CXGate(c,a), TGate(c)])
    #return CXGate(c, b) | SGate(c) | SGate(c) | SGate(c) | TGate(c) | CXGate(c, a) | TGate(c) | CXGate(c, b) | SGate(c) | SGate(c) | SGate(c) | TGate(c) | CXGate(c, a) | TGate(b) | TGate(c) | CXGate(b, a) | TGate(a) | SGate(b) | SGate(b) | SGate(b)| TGate(b) | CXGate(b, a)

def ToffoliGate(a,b,c):
    return HGate(c) | CCZ(a,b,c) | HGate(c)
