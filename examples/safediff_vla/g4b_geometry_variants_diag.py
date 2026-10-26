import sys, json, numpy as np, torch
sys.path.insert(0,'examples/safediff_vla')
from g4b_geometry import *
data=TaskData("select_poker","cpu"); B={}
for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
z=zplane_const(data); Z=torch.load("outputs/train/safediff_vla_g3/M1_s0_preds.pt",weights_only=False)
def cen(m,c):  # mass-weighted pixel centroid in 3x3 nbhd of peak (no GT)
    r,cc=divmod(c,8); rs=range(max(0,r-1),min(8,r+2)); cs=range(max(0,cc-1),min(8,cc+2)); w=np.array([[m[a,b] for b in cs] for a in rs]); ys=np.array([(a+.5)*60 for a in rs]); xs=np.array([(b+.5)*60 for b in cs])
    return (w.sum(0)@xs)/w.sum(), (w.sum(1)@ys)/w.sum()
def proj_px(u,v):
    o,d=ray(CAL[1],u,v); t=(z-o[2])/d[2]; return proj(CAL[0],o+t*d)
for label,sp,hs in (("seen","test",False),("rank9","test_heldout_identity",True)):
    P,W=Z["preds"][sp]; out={}
    for variant in ("cell96","centroid96","cell144","centroid144"):
        side=144 if "144" in variant else 96; hit={1:0,2:0}; n=0
        for j,s in enumerate(data.idx(sp)):
            names=data.names[s]; sid=data.sids[s]
            for k in range(len(names)):
                if not data.valid[s,k] or bool(data.held[s,k])!=hs: continue
                n+=1; tf=B[sid]["boxes_xyxy_per_camera"][names[k]][0]; m=W[j,k,64:128].numpy().reshape(8,8); pf=nms_peaks(W[j,k,:64].numpy().reshape(8,8),2); pr=nms_peaks(m,2)
                fb=[(lambda x,y,s_:(x,y,x+s_,y+s_))(*crop_geom(c,0,480)) for c in pf]; pb=[]
                for c in pr:
                    u,v=cen(m,c) if "centroid" in variant else ((c%8+.5)*60,(c//8+.5)*60); pb.append(front_box_at(*proj_px(u,v),side))
                for K in (1,2): hit[K]+=any(inside(b,tf) for b in fb[:K]+pb[:K])
        print(label,variant,"union recall K1 %.3f K2 %.3f"%(hit[1]/n,hit[2]/n))
