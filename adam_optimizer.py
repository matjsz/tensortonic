import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    mt = beta1 * m + (1-beta1) * grad
    vt = beta2 * v + (1-beta2) * grad**2

    # Temporary and only usedd to compute parameter update
    mhat = mt / (1-beta1**t)
    vhat = vt / (1-beta2**t)

    paramt = param - lr * mhat / (np.sqrt(vhat)+eps)
    
    return paramt, mt, vt
