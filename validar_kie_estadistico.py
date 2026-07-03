#!/usr/bin/env python3
"""
validar_kie_estadistico.py
Validacion estadistica de la diferencia LayoutLMv3 vs LiLT (seccion 4.4 de la tesis).

Carga los DOS modelos guardados (kie_*/best), predice sobre el test (emisores no vistos)
y reporta:
  1) F1 (overall/micro, igual que la metrica del notebook) por modelo.
  2) Intervalos de confianza por BOOTSTRAP (remuestreo de documentos) del F1 de cada
     modelo y de la DIFERENCIA (LiLT - LayoutLMv3). Si el IC de la diferencia no cruza 0,
     la mejora es significativa.
  3) Test de aleatorizacion aproximada (permutacion documento a documento) sobre la diferencia
     de F1; p-valor bilateral (Yeh, 2000). No requiere scipy/statsmodels.

NO re-entrena. Reusa la misma tokenizacion del notebook:
  - layoutlmv3 -> AutoProcessor(apply_ocr=False) con imagen
  - lilt       -> LayoutXLMTokenizerFast (texto+layout, sin imagen)

Uso (en el VPS, venv_kie, con GPU):
  python validar_kie_estadistico.py \
      --dataset_dir ~/agente-documentos/dataset_kie \
      --layoutlmv3  ~/agente-documentos/kie_layoutlmv3/best \
      --lilt        ~/agente-documentos/kie_lilt/best \
      --n_boot 2000
"""
import argparse, json, os, math, random
import numpy as np
import torch
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          LayoutXLMTokenizerFast, AutoModelForTokenClassification)
from PIL import Image
from seqeval.metrics import f1_score

MAX_LEN = 512


def cargar_test(dataset_dir):
    path = os.path.join(dataset_dir, "test.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def cargar_modelo(model_dir, device):
    cfg = AutoConfig.from_pretrained(model_dir)
    usa_imagen = (cfg.model_type == "layoutlmv3")
    if usa_imagen:
        proc = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    else:
        # LiLT: tokenizer texto+layout (mismo vocab XLM-R que se uso al entrenar)
        try:
            proc = LayoutXLMTokenizerFast.from_pretrained(model_dir)
        except Exception:
            proc = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return proc, model, id2label, usa_imagen


@torch.no_grad()
def predecir_doc(ej, proc, model, id2label, usa_imagen, device):
    """Devuelve la secuencia BIO predicha a nivel de PALABRA para un documento."""
    words, boxes = ej["tokens"], ej["bboxes"]
    if usa_imagen:
        img = Image.open(ej["image_path"]).convert("RGB")
        enc = proc(img, words, boxes=boxes, return_tensors="pt",
                   truncation=True, padding="max_length", max_length=MAX_LEN)
    else:
        enc = proc(words, boxes=boxes, return_tensors="pt",
                   truncation=True, padding="max_length", max_length=MAX_LEN)
    word_ids = enc.word_ids(0)
    inputs = {k: v.to(device) for k, v in enc.items()}
    logits = model(**inputs).logits[0]            # (seq, num_labels)
    preds = logits.argmax(-1).tolist()
    # primera subpalabra de cada palabra -> su etiqueta
    word_pred = {}
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in word_pred:
            continue
        word_pred[wid] = id2label[preds[idx]]
    return [word_pred.get(i, "O") for i in range(len(words))]


def spans(seq):
    """Conjunto de entidades (label, ini, fin) a partir de una secuencia BIO."""
    out, i, n = set(), 0, len(seq)
    while i < n:
        t = seq[i]
        if t.startswith("B-"):
            ent = t[2:]; j = i + 1
            while j < n and seq[j] == "I-" + ent:
                j += 1
            out.add((ent, i, j)); i = j
        else:
            i += 1
    return out


def mcnemar_exacto(b, c):
    """p-valor exacto (binomial bilateral) de McNemar sobre pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--layoutlmv3", required=True, help="dir del best de LayoutLMv3")
    ap.add_argument("--lilt", required=True, help="dir del best de LiLT")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    test = cargar_test(args.dataset_dir)
    print("documentos de test:", len(test))

    modelos = {"layoutlmv3": args.layoutlmv3, "lilt": args.lilt}
    gold = [ej["ner_tags"] for ej in test]
    pred = {}
    for nombre, mdir in modelos.items():
        print(f"\nPrediciendo con {nombre} ({mdir}) ...")
        proc, model, id2label, usa_imagen = cargar_modelo(mdir, device)
        pred[nombre] = [predecir_doc(ej, proc, model, id2label, usa_imagen, device) for ej in test]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- 1) F1 overall (micro) por modelo, igual que el notebook ----
    print("\n===== F1 overall (micro) =====")
    f1 = {n: f1_score(gold, pred[n]) for n in modelos}
    for n in modelos:
        print(f"  {n:<12} F1 = {f1[n]:.4f}")
    print(f"  diferencia (lilt - layoutlmv3) = {f1['lilt'] - f1['layoutlmv3']:+.4f}")

    # ---- 2) Bootstrap CIs (remuestreo de documentos) ----
    print(f"\n===== Bootstrap 95% CI ({args.n_boot} resamples, por documento) =====")
    rnd = random.Random(args.seed)
    N = len(test)
    boot = {n: [] for n in modelos}
    boot_diff = []
    for _ in range(args.n_boot):
        idx = [rnd.randrange(N) for _ in range(N)]
        g = [gold[i] for i in idx]
        fs = {}
        for n in modelos:
            p = [pred[n][i] for i in idx]
            fs[n] = f1_score(g, p)
            boot[n].append(fs[n])
        boot_diff.append(fs["lilt"] - fs["layoutlmv3"])

    def ci(v):
        return np.percentile(v, 2.5), np.percentile(v, 97.5)
    for n in modelos:
        lo, hi = ci(boot[n])
        print(f"  {n:<12} F1 = {f1[n]:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
    dlo, dhi = ci(boot_diff)
    signif = "SIGNIFICATIVA (IC no cruza 0)" if (dlo > 0 or dhi < 0) else "no significativa (IC cruza 0)"
    print(f"  diferencia (lilt - layoutlmv3) = {f1['lilt']-f1['layoutlmv3']:+.4f}  IC95% [{dlo:+.4f}, {dhi:+.4f}] -> {signif}")

    # ---- 3) Test de aleatorizacion aproximada (permutacion documento a documento) ----
    print(f"\n===== Aleatorizacion aproximada ({args.n_boot} permutaciones) =====")
    obs = f1["lilt"] - f1["layoutlmv3"]
    rnd2 = random.Random(args.seed + 1); ge = 0
    pa, pb = pred["lilt"], pred["layoutlmv3"]
    for _ in range(args.n_boot):
        A2, B2 = [], []
        for i in range(N):
            if rnd2.random() < 0.5: A2.append(pb[i]); B2.append(pa[i])
            else:                   A2.append(pa[i]); B2.append(pb[i])
        if abs(f1_score(gold, A2) - f1_score(gold, B2)) >= abs(obs): ge += 1
    p_rand = (ge + 1) / (args.n_boot + 1)
    print(f"  diferencia observada |F1(lilt) - F1(layoutlmv3)| = {abs(obs):.4f}")
    print(f"  p-valor (aleatorizacion, bilateral) = {p_rand:.4g}")
    print("  ->", "diferencia significativa (p<0.05)" if p_rand < 0.05 else "sin evidencia de diferencia (p>=0.05)")


if __name__ == "__main__":
    main()
