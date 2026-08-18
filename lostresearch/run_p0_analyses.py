"""
P0 analysis pack v2 (reviewer round 8): endpoint-free taxonomy family,
k-cutoff comparison, full decision tree, Clopper-Pearson CIs.

CIS field semantics (verified against data): stored cis = CIS^gen, i.e.
log p_l(gold) - log p_l(generated first token). Under greedy decoding on
INCORRECT examples the generated token is the final argmax = strongest
non-gold token, so CIS^gen == CIS^comp and Eq. 8 rates are unaffected.
On CORRECT examples the generated first token may differ from the gold
first token (aliases/tokenization), so cis==0 or measures an alias pair;
successful-preservation is therefore defined rank-only for correct cases.

Endpoint-free family: C_rank(k_in, k_out) = 1[min_{l<L} r_l <= k_in] * 1[r_L > k_out].
  (5,0): closest endpoint-free analogue of Eq. 8's final condition (not top-1)
  (4,0): conventional top-5 entry version
  (4,9): reviewer-suggested deep-exit hysteresis (outside top-10)

Usage:  python run_p0_analyses.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def cohen_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return (po - pe) / (pe - 1e-12)


def labels(s, k_form=5):
    rk, cis = s["correct_rank"], s["cis"]
    inter = range(len(rk) - 1)
    formed = any(rk[l] <= k_form and cis[l] > 0 for l in inter)
    return {
        "eq8": int(formed and cis[-1] < 0),
        "formed": int(formed),
        "h50": int(min(rk[:-1]) <= 5 and rk[-1] > 0),
        "h40": int(min(rk[:-1]) <= 4 and rk[-1] > 0),
        "h49": int(min(rk[:-1]) <= 4 and rk[-1] > 9),
        "k4eq8": int(any(rk[l] <= 4 and cis[l] > 0 for l in inter)
                     and cis[-1] < 0),
        "r_final": rk[-1],
        "mid_min": min(rk[:-1]),
    }


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        R = json.load(fh)

    rows = []
    for s in R:
        rk = s.get("correct_rank", [])
        cis = s.get("cis", [])
        if len(rk) < 2 or len(cis) != len(rk):
            continue
        d = labels(s)
        d["task"] = s.get("task", "?")
        d["correct"] = bool(s.get("final_correct"))
        rows.append(d)

    inc = [r for r in rows if not r["correct"]]
    cor = [r for r in rows if r["correct"]]
    tasks = sorted(set(r["task"] for r in inc))
    print(f"n={len(rows)} (correct={len(cor)}, incorrect={len(inc)})\n")

    # ---------- [1] endpoint-free family vs Eq.8 ----------
    print("[1] Endpoint-free taxonomy family vs Eq.8 (incorrect examples)")
    from scipy.stats import spearmanr
    fam = ["h50", "h40", "h49", "k4eq8", "eq8"]
    per_task = {f: {t: 100 * np.mean([r[f] for r in inc if r["task"] == t])
                    for t in tasks} for f in fam}
    print(f"{'task':16s}" + "".join(f"{f:>9s}" for f in fam))
    for t in tasks:
        print(f"{t:16s}" + "".join(f"{per_task[f][t]:8.1f}%" for f in fam))
    print(f"{'OVERALL':16s}" + "".join(
        f"{100*np.mean([r[f] for r in inc]):8.1f}%" for f in fam))
    for f in ["h50", "h40", "h49", "k4eq8"]:
        a = np.array([r[f] for r in inc]); b = np.array([r["eq8"] for r in inc])
        kap = cohen_kappa(a, b)
        jac = (a & b).sum() / max((a | b).sum(), 1)
        rho, _ = spearmanr([per_task[f][t] for t in tasks],
                           [per_task["eq8"][t] for t in tasks])
        print(f"  {f} vs Eq8: kappa={kap:.3f}  Jaccard={jac:.3f}  "
              f"task-order Spearman={rho:.3f}")
    print()

    # ---------- [2] k=4 vs k=5 ----------
    print("[2] Cutoff comparison (CIS condition fixed)")
    print(f"  conventional top-5 (r<5, k=4): "
          f"{100*np.mean([r['k4eq8'] for r in inc]):.1f}%")
    print(f"  stored cutoff (r<=5, k=5):     "
          f"{100*np.mean([r['eq8'] for r in inc]):.1f}%\n")

    # ---------- [3] decision tree ----------
    print("[3] Taxonomy decision tree (mutually exclusive, exhaustive)")
    print("    (correct cases use rank-only formation, see header note)")
    succ = sum(1 for r in cor if r["mid_min"] <= 5)
    late = len(cor) - succ
    form = sum(1 for r in inc if not r["formed"])
    pres = sum(1 for r in inc if r["eq8"])
    unres = sum(1 for r in inc if r["formed"] and not r["eq8"])
    unres_top1 = sum(1 for r in inc if r["formed"] and not r["eq8"]
                     and r["r_final"] == 0)
    print(f"  successful preservation (correct & rank-formed): {succ:5d}")
    print(f"  late-emergent correct  (correct & !rank-formed):  {late:5d}")
    print(f"  formation failure      (inc & !formed):          {form:5d}")
    print(f"  preservation failure   (Eq.8 positive):          {pres:5d}")
    print(f"  unresolved             (formed & final CIS>=0):  {unres:5d} "
          f"(first-token final top-1: {unres_top1})")
    tot = succ + late + form + pres + unres
    print(f"  total {tot} == n {len(rows)}: {tot == len(rows)}")
    print(f"  among errors: formation {100*form/len(inc):.1f}%, "
          f"preservation {100*pres/len(inc):.1f}%, "
          f"unresolved {100*unres/len(inc):.1f}%\n")

    # ---------- [4] Clopper-Pearson ----------
    print("[4] Clopper-Pearson 95% CIs (Eq.8 per task)")
    for t in tasks:
        sub = [r for r in inc if r["task"] == t]
        k_t = sum(r["eq8"] for r in sub)
        lo, hi = clopper_pearson(k_t, len(sub))
        print(f"  {t:16s} {k_t:3d}/{len(sub):4d} = {100*k_t/len(sub):5.1f}%  "
              f"[{100*lo:.1f}, {100*hi:.1f}]")
    k_all = sum(r["eq8"] for r in inc)
    lo, hi = clopper_pearson(k_all, len(inc))
    print(f"  {'OVERALL':16s} {k_all:3d}/{len(inc):4d} = "
          f"{100*k_all/len(inc):5.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")

    out = os.path.join(config.DATA_DIR, "p0_analyses_Qwen3-8B.json")
    with open(out, "w") as fh:
        json.dump({"per_task": per_task,
                   "overall": {f: float(100 * np.mean([r[f] for r in inc]))
                               for f in fam},
                   "taxonomy": {"succ": succ, "late": late, "form": form,
                                "pres": pres, "unres": unres,
                                "unres_top1": unres_top1}}, fh, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
