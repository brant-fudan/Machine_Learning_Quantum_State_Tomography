"""Train a tiny adapted GRU without overwriting results/ from run.py."""
import numpy as np
from quantneg.experiment import make_examples
from quantneg.models import fit_network,predict
if __name__=='__main__':
    rng=np.random.default_rng(123)
    x,y=make_examples(100,'iid',rng,('full','partial'),(64,256))
    model,history=fit_network(x,y,kind='gru',seed=123,epochs=2,batch_size=32)
    out,_=predict(model,x[:4])
    assert np.isfinite(out).all() and (out>=0).all()
    print('adapted-GRU smoke pass; 2 epochs, example Negativities:',out.tolist())
