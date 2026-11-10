import sys, numpy as np, re
sys.argv += []

d = np.load(sys.argv[2]); H = d["H"]
layout = eval(re.search(r"walk sites (\{.*\})", open(sys.argv[3]).read()).group(1))
# rebuild full layout order: need all site sizes -> read from npz? use sizes file
sizes = eval(open(sys.argv[4]).read())
lay, i = {}, 0
for k in sorted(sizes):
    lay[k] = (i, sizes[k]); i += int(np.prod(sizes[k])) if sizes[k] else 1
assert i == H.shape[0], (i, H.shape)
w, V = np.linalg.eigh((H + H.T) / 2)
print("neg eigs", int((w < 0).sum()), "min", w.min(), "max", w.max())
Hp = (V * np.abs(w)) @ V.T
def cond(H):
    w, V = np.linalg.eigh(H)
    cov = (V / w) @ V.T
    s = np.sqrt(np.diag(cov))
    e = np.linalg.eigvalsh(H * s[:, None] * s[None, :])
    return e.max() / e.min(), e
def Q(m):
    L = np.tril(np.ones((m, m))); return np.linalg.eigh(L.T @ L)[1]
def rot(H, sites):
    R = np.eye(H.shape[0])
    for s in sites:
        st, sh = lay[s]; m = sh[0]; k = int(np.prod(sh[1:])) if len(sh) > 1 else 1
        idx = np.arange(st, st + m * k).reshape(m, k)
        for j in range(k):
            R[np.ix_(idx[:, j], idx[:, j])] = Q(m)
    return R.T @ H @ R
walks = [k for k in layout if layout[k][0] > 8]
c0, e0 = cond(Hp); print("current cond", f"{c0:.3g}", "sqrt", f"{np.sqrt(c0):.0f}")
c1, e1 = cond(rot(Hp, walks)); print("rotated all walks", f"{c1:.3g}", "sqrt", f"{np.sqrt(c1):.0f}")
for k in walks:
    c, _ = cond(rot(Hp, [k])); print("  only", k, f"{c:.3g}")
# per-block condition (within block only)
for k in walks:
    st, sh = lay[k]; n = int(np.prod(sh)); b = slice(st, st + n)
    cb, _ = cond(Hp[b, b]); cr, _ = cond(rot(Hp, [k])[b, b])
    print(f"  block {k}: within {cb:.3g} -> rotated {cr:.3g}")
Hr = rot(Hp, walks)
w, V = np.linalg.eigh(Hr); cov = (V / w) @ V.T; s = np.sqrt(np.diag(cov))
Hs = Hr * s[:, None] * s[None, :]
e, U = np.linalg.eigh(Hs)
names = np.empty(H.shape[0], dtype=object)
for k, (st, sh) in lay.items():
    n = int(np.prod(sh)) if sh else 1; names[st:st + n] = k
def show(j):
    u = U[:, j] ** 2
    top = {}
    for k in set(names):
        top[k] = u[names == k].sum()
    best = sorted(top.items(), key=lambda t: -t[1])[:4]
    print(f"  eig {e[j]:.3g}:", ", ".join(f"{k} {v:.2f}" for k, v in best))
print("smallest"); [show(j) for j in range(4)]
print("largest"); [show(j) for j in range(len(e) - 4, len(e))]
groups = {
 "rt": [k for k in lay if k.startswith("log_rt") or k == "initial_infections"],
 "pi": [k for k in lay if k.startswith("logit_pi0") or k.startswith("missing_fraction")],
 "dshape": ["delay_hazard0", "delay_shape_z"],
 "dtime": [k for k in lay if k.startswith("delay_time") or k.startswith("delay_offset")],
 "asc": [k for k in lay if k.startswith("ascertainment")],
}
def idx_of(sites):
    out = []
    for k in sites:
        st, sh = lay[k]; n = int(np.prod(sh)) if sh else 1; out += list(range(st, st + n))
    return np.array(out)
def block_whiten(H, gs):
    R = np.eye(H.shape[0])
    for g in gs:
        ii = idx_of(groups[g]); _, Qb = np.linalg.eigh(H[np.ix_(ii, ii)]); R[np.ix_(ii, ii)] = Qb
    return R.T @ H @ R
for g in groups: print(g, groups[g])
c, _ = cond(block_whiten(Hp, list(groups))); print("block-eigen rotate all groups (bound):", f"{c:.3g}", "sqrt", f"{np.sqrt(c):.0f}")
for g in groups:
    c, _ = cond(block_whiten(Hp, [g])); print("  only", g, f"{c:.3g}")
c, _ = cond(block_whiten(rot(Hp, walks), ["rt", "pi"])); print("walk rotation + rt,pi groups:", f"{c:.3g}")
