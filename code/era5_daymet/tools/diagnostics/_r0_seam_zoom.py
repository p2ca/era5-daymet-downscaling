"""R0 阶梯感特写: 跨 tile 边界放大, Truth vs R0 并排。若有阶梯, 缝隙线两侧会突变。
每变量两行: 整场(叠青色 tile 边界) + 跨竖直边界的放大窗(Truth|R0)。英文标题避免字体缺字。"""
# Packaged implementation; the original code/ path remains compatible.
import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as VM

W="/lustre/orion/atm112/scratch/hjsong/downscaling"
CK=f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/ckpt.pt"; DEV="cuda"
DAY=int(sys.argv[1]) if len(sys.argv)>1 else 200
a=torch.load(CK,map_location="cpu")["args"]; iv,ov=a["in_vars"],a["out_vars"]
stats=TD.Stats(a["stats_dir"],iv,ov)
test=TD.DownscaleData(a["era5_dir"],a["daymet_dir"],[a["test_year"]],iv,ov,stats)
Cin,Cout=len(iv)+3+len(ov),len(ov); tile=a["patch"]; step=tile-max(tile//4,1)
net=VM.ViT(Cin,Cout,img=tile,patch=a["vit_patch"],dim=a["dim"],depth=a["depth"],heads=a["heads"],mlp=a["mlp"]).to(DEV)
net.load_state_dict(torch.load(CK,map_location=DEV)["model"]); net.eval()

cond,_,m,hr=test.full(a["test_year"],DAY); land=(m[0] if m.ndim==3 else m)>0.5
with torch.no_grad():
    o=TD.det_predict(net,torch.from_numpy(cond[None]).float().to(DEV),tile,DEV)
pred=o*stats.d_std[:,None,None]+stats.d_mean[:,None,None]
if "total_precipitation_24hr" in ov:
    pi=ov.index("total_precipitation_24hr")
    pred[pi]=TD.precip_inv(pred[pi],stats.precip_scale) if stats.precip_log else np.maximum(pred[pi],0)
    hr=hr.copy()  # hr 已是物理量
H,Wd=land.shape
vbnds=[b for b in range(step,Wd,step)]                       # 竖直 tile 边界列
# 选一个"两侧都主要是陆地"的竖直边界做放大窗
def land_frac(xb):
    y0,y1=H//4,3*H//4; x0,x1=max(0,xb-25),min(Wd,xb+25)
    return land[y0:y1,x0:x1].mean(),(y0,y1,x0,x1)
xb=max(vbnds,key=lambda b:land_frac(b)[0]); frac,(y0,y1,x0,x1)=land_frac(xb)
print(f"[zoom] day={DAY} 选竖直边界 x={xb} (窗内陆地比 {frac:.0%})  tile 边界列={vbnds}",flush=True)

specs=[("2m_temperature_max","tmax [C-ish, norm-denorm]","RdYlBu_r"),
       ("total_precipitation_24hr","precip [mm/day]","viridis")]
fig,axes=plt.subplots(len(specs),3,figsize=(16,5*len(specs)))
if len(specs)==1: axes=axes[None]
for r,(vn,lab,cm) in enumerate(specs):
    if vn not in ov: continue
    vi=ov.index(vn); P=np.where(land,pred[vi],np.nan); T=np.where(land,hr[vi],np.nan)
    if vn.startswith("total"):  # precip 用 log 拉伸显示
        P=np.log1p(np.clip(P,0,None)); T=np.log1p(np.clip(T,0,None)); lab="precip log1p(mm/day)"
    vmn,vmx=np.nanpercentile(T,2),np.nanpercentile(T,98)
    ax=axes[r]
    ax[0].imshow(P,cmap=cm,vmin=vmn,vmax=vmx); ax[0].set_title(f"R0 {lab} full (day{DAY})")
    for b in vbnds: ax[0].axvline(b,color="cyan",lw=0.5,alpha=0.6)
    for b in range(step,H,step): ax[0].axhline(b,color="cyan",lw=0.5,alpha=0.6)
    ax[0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,ec="lime",fc="none",lw=1.5))
    tz=T[y0:y1,x0:x1]; pz=P[y0:y1,x0:x1]; xl=xb-x0
    ax[1].imshow(tz,cmap=cm,vmin=vmn,vmax=vmx); ax[1].set_title("Truth  zoom @ seam")
    ax[1].axvline(xl,color="cyan",lw=1.0,alpha=0.8)
    ax[2].imshow(pz,cmap=cm,vmin=vmn,vmax=vmx); ax[2].set_title("R0 ViT  zoom @ seam (staircase? look across cyan line)")
    ax[2].axvline(xl,color="cyan",lw=1.0,alpha=0.8)
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
plt.tight_layout()
out=f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/r0_seam_zoom_day{DAY}.png"
plt.savefig(out,dpi=150,bbox_inches="tight"); print(f"[zoom] saved {out}",flush=True)
