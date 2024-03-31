import numpy as np
from copy import deepcopy

def fmi_bost_0_rows_fn(h5, measurement, sid, sent=None):
    base_measurement = np.abs(h5[measurement][str(sid)][:]) 
    fill_mask = np.logical_not(deepcopy(sent.uposmask)) # True -> already filled, False -> not filled
    max_len = base_measurement.shape[0]
    heuristic_value = 1e-2
    
    for i in range(1, max_len):
        fill_mask = fill_mask[:-1]
        # if i>1:
        base_measurement_nk = np.diag(np.where(np.abs(h5[measurement][str(sid)][:]).sum(1)<heuristic_value, heuristic_value, 0)[:-i] * (sent.uposmask[i:])* (sent.uposmask[:-i]) * np.logical_not(fill_mask), i)
        heuristic_value *= 0.9
        fill_mask = np.logical_or((np.logical_not(sent.uposmask[i:])* np.logical_not(sent.uposmask[:-i])).astype(bool), fill_mask)
        base_measurement += base_measurement_nk
        
    
    return base_measurement# + base_measurement_n1 + base_measurement_n2 + base_measurement_n3 + base_measurement_n4
    

    # np.diag(np.where(np.abs(h5[measurement][str(sid)][:]).sum(1)<1e-2, 1e-2, 0)[:-1] * sent.uposmask[1:]* sent.uposmask[:-1]
fmi_retrival_fn = lambda h5, measurement, sid, sent=None: np.abs(h5[measurement][str(sid)][:])
pmi_retrival_fn = lambda h5, measurement, sid, sent=None: np.triu(h5[measurement][str(sid)][:] + np.transpose(h5[measurement][str(sid)][:]))
pmi_abs_fn = lambda h5, measurement, sid, sent=None: np.triu(np.abs(h5[measurement][str(sid)][:]) + np.abs(np.transpose(h5[measurement][str(sid)][:])))