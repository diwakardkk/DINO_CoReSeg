"""Native-grid binary metrics. Surface failure penalties are explicit in pixels."""
import numpy as np
from scipy import ndimage as ndi


def surface(mask):return mask & ~ndi.binary_erosion(mask,border_value=0)


def positive_metrics(prob,truth,threshold=.5,tolerance=2,band_fraction=.02):
    prob=np.asarray(prob,float);truth=np.asarray(truth,bool);pred=prob>=threshold
    if prob.shape!=truth.shape or not np.isfinite(prob).all():raise ValueError('Invalid metric arrays')
    if not truth.any():raise ValueError('Use normal_metrics for empty target')
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn)
    r={'dice':2*tp/(2*tp+fp+fn),'iou':tp/(tp+fp+fn),'precision':precision,'recall':recall,
       'specificity':tn/(tn+fp) if tn+fp else None,'f2':5*tp/(5*tp+4*fn+fp),
       'mae':float(np.abs(prob-truth).mean()),'empty_prediction':int(not pred.any())}
    if not pred.any():
        penalty=float(np.hypot(*truth.shape))
        r.update(boundary_f1=0.,boundary_iou=0.,hd95=penalty,assd=penalty,surface_failure=1)
        return r
    ps,ts=surface(pred),surface(truth)
    dtp=ndi.distance_transform_edt(~ps);dtt=ndi.distance_transform_edt(~ts)
    bp=float((dtt[ps]<=tolerance).mean());br=float((dtp[ts]<=tolerance).mean())
    r['boundary_f1']=2*bp*br/(bp+br) if bp+br else 0.
    radius=max(1,int(round(band_fraction*np.hypot(*truth.shape))))
    pb=pred&~ndi.binary_erosion(pred,iterations=radius,border_value=0)
    tb=truth&~ndi.binary_erosion(truth,iterations=radius,border_value=0)
    r['boundary_iou']=float((pb&tb).sum()/(pb|tb).sum())
    distances=np.concatenate([dtt[ps],dtp[ts]])
    r.update(hd95=float(np.percentile(distances,95)),assd=float(distances.mean()),surface_failure=0)
    return r


def normal_metrics(prob,threshold=.5,gates=(1,10,50,100,500)):
    prob=np.asarray(prob,float);pred=prob>=threshold
    if prob.ndim!=2 or not np.isfinite(prob).all():raise ValueError('Invalid normal probability map')
    labels,n=ndi.label(pred,structure=np.ones((3,3)))
    areas=np.bincount(labels.ravel())[1:];maximum=int(areas.max()) if len(areas) else 0
    result={'fp_frame':int(pred.any()),'frame_specificity':float(not pred.any()),
            'pixel_specificity':float((~pred).mean()),'fp_area_pixels':int(pred.sum()),
            'foreground_fraction':float(pred.mean()),'fp_components':int(n),'mae':float(prob.mean())}
    for gate in gates:
        result[f'fp_frame_gate_{gate}']=int(maximum>=gate)
        result[f'frame_specificity_gate_{gate}']=float(maximum<gate)
    return result


def macro(rows):
    keys=sorted(set().union(*(r.keys() for r in rows))) if rows else []
    return {k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys
            if all(isinstance(r.get(k),(int,float,type(None))) for r in rows) and any(r.get(k) is not None for r in rows)}
