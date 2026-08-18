"""
Decoder-stability of the taxonomy (reviewer round 8, P0-2).

For the same errors, compute taxonomy labels under three decoders:
  raw lens   : intermediate hidden -> final RMSNorm + unembedding
  tuned lens : per-layer affine translator (trained on 200 unlabelled
               prompts toward final logits) -> final norm + unembedding
  cosine     : rank by cosine(h_l, w_y) over the vocabulary (parameter-free)

Labels compared:
  h50  (endpoint-free rank): 1[min_{l<L} r_l <= 5] * 1[r_L > 0]
                              -- computable under ALL three decoders
  eq8  (needs probabilities): rank<=5 & margin>0 intermediate, final
                              margin<0 -- computed for raw and tuned only
                              (cosine has no probability semantics)

Outputs the reviewer-requested table: per-decoder rate, Cohen's kappa and
Jaccard vs tuned, per-task agreement, and a 2-of-3 consensus rate.

Usage:
  python run_decoder_stability.py --n 500
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples
from run_p0_tuned_lens import collect_all_hiddens


def train_translators_mb(model, hiddens, final_logits, num_layers,
                         epochs=150, batch=32, lr=1e-3):
    """Memory-friendly tuned-lens trainer (mini-batch, cached fp32 unembed).

    The original trainer materializes a fresh fp32 unembedding per layer and
    backprops over the full 200-prompt batch at once, which OOMs a 24GB GPU
    next to the loaded model. This version caches one fp32 weight copy and
    steps in batches of 32; affine fitting converges well within 150 epochs.
    """
    device = model.device
    final_norm = model.model.norm
    unembed_f = model.lm_head.weight.detach().float()  # [V, d], once
    translators = {}
    for l in range(num_layers):
        h = hiddens[l].to(device).float()       # [n, d]
        y = final_logits.to(device).float()     # [n, V]
        A = torch.eye(h.shape[1], device=device)
        b = torch.zeros(h.shape[1], device=device)
        A.requires_grad_(True)
        b.requires_grad_(True)
        opt = torch.optim.Adam([A, b], lr=lr)
        n = h.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                normed = final_norm(h[idx] @ A.T + b)      # [B, d]
                logits = F.linear(normed, unembed_f)       # [B, V]
                loss = F.mse_loss(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        if l % 6 == 0:
            print(f"  translator layer {l}: final loss {loss.item():.4f}")
        translators[l] = {"A": A.detach().cpu(), "b": b.detach().cpu()}
        del h, y
        torch.cuda.empty_cache()
    return translators


def cohen_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (pe - 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="errors to decode (use 1494 for full set)")
    parser.add_argument("--lens-prompts", type=int, default=200)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as f:
        all_results = json.load(f)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = list(model.model.layers)
    num_layers = len(layers)
    final_norm = model.model.norm
    unembed = model.lm_head.weight.float()          # [V, d]
    unembed_n = F.normalize(unembed, dim=1)         # for cosine
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}
    errors = [s for s in all_results if not s.get("final_correct")]
    errors = [s for s in errors if s["id"] in prep_map][:args.n]
    print(f"Decoding {len(errors)} errors under three decoders")

    # ---------- train tuned lens (memory-friendly) ----------
    print(f"\nTraining tuned-lens translators on {args.lens_prompts} prompts...")
    lens_samples = [s for s in prepared if s["id"] not in
                    {e["id"] for e in errors}][:args.lens_prompts]
    hiddens, final_logits, _ = collect_all_hiddens(model, lens_samples,
                                                   n_samples=len(lens_samples))
    translators = train_translators_mb(model, hiddens, final_logits,
                                       num_layers)
    del hiddens, final_logits
    torch.cuda.empty_cache()
    print("Translators ready.")

    # ---------- decode ----------
    def decode_all(sample):
        """Return per-layer ranks and margins under 3 decoders."""
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long,
                                 device=device)
        gold_token = sample["primary_answer_ids"][0]
        gen_text = next((e["generated"] for e in errors
                         if e["id"] == sample["id"]), "")
        gen_ids = tokenizer.encode(gen_text, add_special_tokens=False) \
            if gen_text else []
        gen_tok = gen_ids[0] if gen_ids else gold_token

        cap = {}
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            cap.setdefault("hs", []).append(h[0, -1, :].detach().float())
        handles = [l.register_forward_hook(hook) for l in layers]
        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in handles:
            h.remove()
        hs = torch.stack(cap["hs"])                # [L, d]

        out = {}
        model_dtype = next(model.lm_head.parameters()).dtype
        with torch.no_grad():
            for name in ["raw", "tuned"]:
                ranks, margins = [], []
                for l in range(num_layers):
                    h = hs[l].unsqueeze(0)
                    if name == "tuned":
                        t = translators[l]
                        h = h @ t["A"].to(device) + t["b"].to(device)
                    h = h.to(model_dtype)
                    lg = model.lm_head(final_norm(h))[0].float()
                    lp = F.log_softmax(lg, dim=-1)
                    ranks.append(int((lg > lg[gold_token]).sum()))
                    margins.append(float(lp[gold_token] - lp[gen_tok]))
                out[name] = {"ranks": ranks, "margins": margins}
            # cosine decoder (rank only)
            ranks = []
            hn = F.normalize(hs, dim=1)            # [L, d]
            for l in range(num_layers):
                sims = hn[l] @ unembed_n.T         # [V]
                ranks.append(int((sims > sims[gold_token]).sum()))
            out["cosine"] = {"ranks": ranks}
        return out

    def h50(ranks):
        return int(min(ranks[:-1]) <= 5 and ranks[-1] > 0)

    def eq8(ranks, margins):
        inter = range(len(ranks) - 1)
        formed = any(ranks[l] <= 5 and margins[l] > 0 for l in inter)
        return int(formed and margins[-1] < 0)

    recs = []
    for e in tqdm(errors, desc="decode"):
        s = prep_map[e["id"]]
        d = decode_all(s)
        rec = {"id": e["id"], "task": e.get("task", "?")}
        for name in ["raw", "tuned", "cosine"]:
            rec[f"h50_{name}"] = h50(d[name]["ranks"])
        for name in ["raw", "tuned"]:
            rec[f"eq8_{name}"] = eq8(d[name]["ranks"], d[name]["margins"])
        recs.append(rec)

    # ---------- report ----------
    print("\n" + "=" * 70)
    print(f"Decoder stability (n={len(recs)})")
    print("=" * 70)
    print(f"\n{'decoder':10s} {'h50 rate':>9s} {'eq8 rate':>9s} "
          f"{'kappa(h50) vs tuned':>20s} {'Jaccard(h50)':>13s}")
    for name in ["raw", "tuned", "cosine"]:
        h = np.array([r[f"h50_{name}"] for r in recs])
        ht = np.array([r["h50_tuned"] for r in recs])
        row = f"{name:10s} {100*h.mean():8.1f}%"
        if f"eq8_{name}" in recs[0]:
            q = np.array([r[f"eq8_{name}"] for r in recs])
            row += f" {100*q.mean():8.1f}%"
        else:
            row += f" {'n/a':>9s}"
        if name != "tuned":
            kap = cohen_kappa(h, ht)
            jac = (h & ht).sum() / max((h | ht).sum(), 1)
            row += f" {kap:20.3f} {jac:13.3f}"
        print(row)
    # eq8 raw vs tuned
    qr = np.array([r["eq8_raw"] for r in recs])
    qt = np.array([r["eq8_tuned"] for r in recs])
    print(f"\nEq.8 raw vs tuned: kappa={cohen_kappa(qr, qt):.3f}, "
          f"Jaccard={(qr & qt).sum()/max((qr | qt).sum(),1):.3f}, "
          f"rates {100*qr.mean():.1f}% vs {100*qt.mean():.1f}%")
    # consensus 2-of-3 on h50
    votes = (np.array([r["h50_raw"] for r in recs]) +
             np.array([r["h50_tuned"] for r in recs]) +
             np.array([r["h50_cosine"] for r in recs]))
    print(f"Consensus 2-of-3 (h50): {100*np.mean(votes >= 2):.1f}%")
    # per-task tuned h50
    print("\nPer-task h50 (tuned):")
    for t in sorted(set(r["task"] for r in recs)):
        sub = [r for r in recs if r["task"] == t]
        print(f"  {t:16s} {100*np.mean([r['h50_tuned'] for r in sub]):5.1f}%  "
              f"(n={len(sub)})")

    out = os.path.join(config.DATA_DIR, "decoder_stability_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"n": len(recs), "records": recs}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
