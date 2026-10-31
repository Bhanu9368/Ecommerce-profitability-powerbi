"""Generates the sample e-commerce orders dataset (data/ecommerce_orders.csv)."""
import numpy as np, pandas as pd, calendar
from scipy.optimize import linprog

rng = np.random.default_rng(42)
CATS = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Others"]
CHS  = ["Social Media", "Search Ads", "Email", "Direct", "Affiliates", "Referral"]
REV_TOTAL, COST_TOTAL = 1248930, 826540
PROFIT_TOTAL = REV_TOTAL - COST_TOTAL            # 422,390

rev_share = [.28, .22, .18, .12, .10, .10]
cat_rev = [round(REV_TOTAL * s) for s in rev_share]; cat_rev[-1] += REV_TOTAL - sum(cat_rev)
cat_profit = [123200, 93500, 73800, 53600, 43700, 34590]
ch_profit  = [114000, 92900, 74500, 63500, 33800, 43690]
assert sum(cat_profit) == PROFIT_TOTAL and sum(ch_profit) == PROFIT_TOTAL

# Fixed cells shown in the "Category & Channel Wise Profitability" table
FIXED = {(0,0):(250000,90000), (1,1):(190000,70000), (2,2):(150000,55000),
         (3,3):(100000,35000), (4,4):(80000,28000)}

nC, nH = len(CATS), len(CHS); N = nC*nH
idx = lambda c,h: c*nH+h
# variables: R[36], P[36], d[36] (abs deviation of R from prior)
prior = np.zeros(N)
for c in range(nC):
    w = rng.uniform(.6,1.4,nH); w/=w.sum()
    for h in range(nH): prior[idx(c,h)] = cat_rev[c]*w[h]
cost_vec = np.r_[np.zeros(2*N), np.ones(N)]
A_eq, b_eq = [], []
def row(): return np.zeros(3*N)
for c in range(nC):                       # category revenue
    r=row(); r[[idx(c,h) for h in range(nH)]]=1; A_eq.append(r); b_eq.append(cat_rev[c])
    r=row(); r[[N+idx(c,h) for h in range(nH)]]=1; A_eq.append(r); b_eq.append(cat_profit[c])
for h in range(nH):                       # channel profit
    r=row(); r[[N+idx(c,h) for c in range(nC)]]=1; A_eq.append(r); b_eq.append(ch_profit[h])
for (c,h),(R,P) in FIXED.items():
    r=row(); r[idx(c,h)]=1; A_eq.append(r); b_eq.append(R)
    r=row(); r[N+idx(c,h)]=1; A_eq.append(r); b_eq.append(P)
A_ub, b_ub = [], []
for k in range(N):                        # margin bounds 22%-40%
    r=row(); r[N+k]=1; r[k]=-.40; A_ub.append(r); b_ub.append(0)
    r=row(); r[N+k]=-1; r[k]=.22; A_ub.append(r); b_ub.append(0)
    r=row(); r[k]=1; r[2*N+k]=-1; A_ub.append(r); b_ub.append(prior[k])
    r=row(); r[k]=-1; r[2*N+k]=-1; A_ub.append(r); b_ub.append(-prior[k])
bounds=[]
for k in range(N):
    c,h=divmod(k,nH)
    bounds.append((250000,250000) if False else ((FIXED[(c,h)][0],)*2 if (c,h) in FIXED else (2500,79000) if True else None))
bounds += [(0,None)]*N + [(0,None)]*N
res = linprog(cost_vec, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
assert res.success, res.message
R = np.round(res.x[:N]).astype(int); P = np.round(res.x[N:2*N]).astype(int)
# integer fix-ups on non-fixed cells so every row/col sum is exact
free = [k for k in range(N) if divmod(k,nH) not in FIXED]
for c in range(nC):
    ks=[idx(c,h) for h in range(nH)]
    R[[k for k in ks if k in free][0]] += cat_rev[c]-R[ks].sum()
for c in range(nC):
    ks=[idx(c,h) for h in range(nH)]
    P[[k for k in ks if k in free][-1]] += cat_profit[c]-P[ks].sum()
for h in range(nH):
    ks=[idx(c,h) for c in range(nC)]
    d=ch_profit[h]-P[ks].sum()
    if d:                                  # move d between two rows inside this channel column
        fk=[k for k in ks if k in free]; P[fk[0]]+=d; P[fk[1]]-=d
        # compensate rows
        # (fk[0],fk[1] are in different category rows -> rebalance in another channel)
        for kk,sign in ((fk[0],-1),(fk[1],1)):
            c=kk//nH; other=[idx(c,x) for x in range(nH) if idx(c,x) in free and idx(c,x)!=kk and x!=h][0]
            P[other]+=sign*d
# remaining channel mismatch check
chk=[P[[idx(c,h) for c in range(nC)]].sum()-ch_profit[h] for h in range(nH)]
rowchk=[P[[idx(c,h) for h in range(nH)]].sum()-cat_profit[c] for c in range(nC)]
print("channel-profit diffs",chk,"category-profit diffs",rowchk)

# ---- monthly split ----
rev_shape = np.array([45,60,90,75,95,100,115,105,120,155,135,170],float)
prof_shape = np.array([15,25,30,30,40,45,55,60,70,65,75,80],float)
def lr_split(total, w):
    w=np.asarray(w,float); w=w/w.sum(); raw=w*total
    out=np.floor(raw).astype(int); rem=int(total-out.sum())
    for i in np.argsort(-(raw-out))[:rem]: out[i]+=1
    return out
PRODUCTS = {
 "Electronics":["Wireless Earbuds","Smartwatch","Bluetooth Speaker","Power Bank","Phone Case"],
 "Fashion":["Men's T-Shirt","Women's Kurti","Sneakers","Denim Jeans","Handbag"],
 "Home & Kitchen":["Air Fryer","Mixer Grinder","Bedsheet Set","Water Bottle","Storage Box"],
 "Beauty":["Face Serum","Sunscreen","Lipstick","Hair Oil","Perfume"],
 "Sports":["Yoga Mat","Dumbbell Set","Cricket Bat","Football","Resistance Bands"],
 "Others":["Notebook Pack","Gift Card","Desk Organizer","Stationery Kit","Car Accessory"]}
AVG = {"Electronics":4000,"Fashion":1500,"Home & Kitchen":2200,"Beauty":900,"Sports":1400,"Others":800}
REGIONS=["North","South","East","West"]; RW=[.32,.26,.18,.24]

rows=[]
for c,cat in enumerate(CATS):
    for h,ch in enumerate(CHS):
        k=idx(c,h)
        mr = lr_split(R[k], rev_shape*rng.uniform(.9,1.1,12))
        mp = lr_split(P[k], prof_shape*rng.uniform(.9,1.1,12))
        for m in range(12):
            rv,pf=int(mr[m]),int(mp[m])
            if pf>=rv: pf=int(rv*.3)
            n=max(1,int(round(rv/AVG[cat])))
            rs=lr_split(rv, rng.dirichlet(np.ones(n)*4)+1e-9)
            ps=lr_split(pf, rs/ rs.sum())
            for i in range(n):
                rows.append(dict(Category=cat,Channel=ch,Month=m+1,Revenue=int(rs[i]),Profit=int(ps[i])))
df=pd.DataFrame(rows)
df["Cost"]=df.Revenue-df.Profit
df=df[df.Revenue>0].copy()
days=[calendar.monthrange(2024,m)[1] for m in df.Month]
df["Date"]=[pd.Timestamp(2024,m,int(rng.integers(1,d+1))) for m,d in zip(df.Month,days)]
df["Product Name"]=[rng.choice(PRODUCTS[c]) for c in df.Category]
df["Region"]=rng.choice(REGIONS,len(df),p=RW)
avgp=df.Category.map(AVG)
df["Quantity"]=np.maximum(1,np.round(df.Revenue/ (avgp/ rng.uniform(1,2.5,len(df))))).astype(int)
df=df.sort_values("Date").reset_index(drop=True)

# ---- 2023 (previous period): revenue -> +18% YoY, profit -> +28% YoY ----
rev23_t = round(REV_TOTAL/1.18); prof23_t = round(PROFIT_TOTAL/1.28)
d23=df.copy()
d23["Revenue"]=np.maximum(1,np.round(df.Revenue*rev23_t/REV_TOTAL)).astype(int)
d23["Profit"]=np.round(df.Profit*prof23_t/PROFIT_TOTAL).astype(int)
d23.loc[d23.Revenue.idxmax(),"Revenue"]+=rev23_t-d23.Revenue.sum()
d23.loc[d23.Profit.idxmax(),"Profit"]+=prof23_t-d23.Profit.sum()
d23["Cost"]=d23.Revenue-d23.Profit
d23["Date"]=d23.Date.apply(lambda x:x.replace(year=2023,day=min(x.day,28)) if x.month==2 else x.replace(year=2023))

full=pd.concat([d23,df],ignore_index=True).sort_values("Date").reset_index(drop=True)
full.insert(0,"Order ID",[f"ORD{100000+i}" for i in range(len(full))])
full["Date"]=full.Date.dt.strftime("%Y-%m-%d")
full=full[["Order ID","Date","Category","Product Name","Marketing Channel" if False else "Channel","Region","Quantity","Revenue","Cost"]]
full=full.rename(columns={"Channel":"Marketing Channel"})
full.to_csv("data/ecommerce_orders.csv",index=False)
print(len(full),"rows")
