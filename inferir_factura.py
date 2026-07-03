#!/usr/bin/env python3
"""
inferir_factura.py
Inferencia KIE sobre facturas NUEVAS usando modelo(s) YA entrenado(s) (kie_*/best),
sin re-entrenar. Lee el/los JSON de tokens que produce extraer_facturas.py (Paso 1),
predice las etiquetas BIO, agrupa los spans y devuelve los campos extraidos
(+ una limpieza por regex de los tokens fundidos rotulo+valor).

Modelo final de produccion: ENSAMBLE FOCAL = kie_lilt_focal/best + kie_layoutlmv3_focal/best
(promedio de probabilidades por palabra, peso 0.5). Es el de MEJOR desempeno del estudio
(F1 micro 0,755; completitud por documento 69,1 %). Requiere ejecutar dos modelos (uno con
rama visual sobre la imagen) -> GPU recomendada. Ver tesis §4.11.
El modo de un solo modelo (--modelo, sin --modelo2) queda disponible como alternativa ligera.

Flujo de produccion (2 pasos, por los venv separados):
  1) OCR  (venv de PaddleOCR):  python extraer_facturas.py --facturas_dir ... --salida_dir facturas_ocr
  2) KIE  (venv_kie, este script):
       # ENSAMBLE FOCAL (produccion):
       python inferir_factura.py --ocr_dir facturas_ocr \
              --modelo kie_lilt_focal/best --modelo2 kie_layoutlmv3_focal/best --peso 0.5 --salida pred_kie
       # un solo modelo (alternativa):
       python inferir_factura.py --ocr_dir facturas_ocr --modelo kie_lilt/best --salida pred_kie

Nota: para LayoutLMv3 (kie_layoutlmv3*/best) el JSON debe traer 'imagen' con la ruta del PNG.
"""
import argparse, glob, json, os, re
import numpy as np
import torch
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          LayoutXLMTokenizerFast, AutoModelForTokenClassification)

MAX_LEN = 512


def _norm_bbox(box, W, H):
    def c(v, lim):
        return min(1000, max(0, int(round(v / lim * 1000))))
    return [c(box[0], W), c(box[1], H), c(box[2], W), c(box[3], H)]


def cargar_modelo(model_dir, device):
    cfg = AutoConfig.from_pretrained(model_dir)
    usa_imagen = (cfg.model_type == "layoutlmv3")
    if usa_imagen:
        proc = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    else:
        try:
            proc = LayoutXLMTokenizerFast.from_pretrained(model_dir)
        except Exception:
            proc = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return proc, model, id2label, usa_imagen


def _encode(words, boxes_norm, proc, usa_imagen, imagen_path):
    if usa_imagen:
        from PIL import Image
        img = Image.open(imagen_path).convert("RGB")
        return proc(img, words, boxes=boxes_norm, return_tensors="pt",
                    truncation=True, padding="max_length", max_length=MAX_LEN)
    return proc(words, boxes=boxes_norm, return_tensors="pt",
                truncation=True, padding="max_length", max_length=MAX_LEN)


@torch.no_grad()
def predecir_labels(words, boxes_norm, proc, model, id2label, usa_imagen, device, imagen_path=None):
    """Etiqueta BIO por palabra (modelo unico, argmax)."""
    enc = _encode(words, boxes_norm, proc, usa_imagen, imagen_path)
    word_ids = enc.word_ids(0)
    inputs = {k: v.to(device) for k, v in enc.items()}
    preds = model(**inputs).logits[0].argmax(-1).tolist()
    wp = {}
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in wp:
            continue
        wp[wid] = id2label[preds[idx]]
    return [wp.get(i, "O") for i in range(len(words))]


@torch.no_grad()
def predecir_probs(words, boxes_norm, proc, model, usa_imagen, device, LABELS, imagen_path=None):
    """Probs softmax por palabra, reordenadas al orden comun de LABELS (para promediar 2 modelos)."""
    enc = _encode(words, boxes_norm, proc, usa_imagen, imagen_path)
    word_ids = enc.word_ids(0)
    inputs = {k: v.to(device) for k, v in enc.items()}
    p = torch.softmax(model(**inputs).logits[0], -1).cpu().numpy()
    col = {l: int(i) for l, i in model.config.label2id.items()}
    P = np.zeros((len(words), len(LABELS))); seen = set()
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid); P[wid] = [p[idx, col[l]] for l in LABELS]
    return P


def extraer_spans(words, labels):
    campos, i, n = {}, 0, len(labels)
    while i < n:
        t = labels[i]
        if t.startswith("B-"):
            ent = t[2:]; j = i + 1
            while j < n and labels[j] == "I-" + ent:
                j += 1
            campos.setdefault(ent, []).append(" ".join(words[i:j]).strip())
            i = j
        else:
            i += 1
    return campos


def limpiar(entidad, texto):
    if entidad in ("NIT_EMISOR", "NIT_ADQUIRIENTE"):
        m = re.search(r"\d[\d.\s]{5,}\d(?:\s*-\s*\d)?", texto)
        return re.sub(r"[.\s]", "", m.group(0)) if m else texto
    if entidad in ("VALOR_TOTAL", "IVA_TOTAL"):
        m = re.search(r"\d[\d.,\s]*\d", texto)
        return re.sub(r"[^\d]", "", m.group(0)) if m else texto
    if entidad == "NUMERO_FACTURA":
        m = re.search(r"[A-Za-z]{0,6}[-\s]?\d{2,}[-\s]?\d*", texto)
        return m.group(0).replace(" ", "") if m else texto
    if entidad == "FECHA_GENERACION":
        m = re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{8}", texto)
        return m.group(0) if m else texto
    return texto


def procesar_json(jpath, predictor, limpiar_regex=True):
    """predictor(words, boxes_norm, imagen_path) -> lista de etiquetas BIO por palabra."""
    fac = json.loads(open(jpath, encoding="utf-8").read())
    words = [t["text"] for t in fac.get("tokens", [])]
    W, H = fac.get("img_width"), fac.get("img_height")
    out = {"id": os.path.splitext(os.path.basename(jpath))[0],
           "cufe": fac.get("cufe"), "qr_raw": fac.get("qr_raw")}
    if not words or not W or not H:
        out["error"] = "sin tokens o sin img_width/height"
        return out
    boxes = [_norm_bbox(t["bbox"], W, H) for t in fac["tokens"]]
    labels = predictor(words, boxes, fac.get("imagen"))
    spans = extraer_spans(words, labels)
    campos = {}
    for ent, lista in spans.items():
        campos[ent] = limpiar(ent, lista[0]) if limpiar_regex else lista[0]
        if len(lista) > 1:
            campos[ent + "_alt"] = lista[1:]
    out["campos"] = campos
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ocr_json", help="Un JSON de extraer_facturas.py")
    g.add_argument("--ocr_dir", help="Carpeta con JSON de extraer_facturas.py")
    ap.add_argument("--modelo", default="kie_lilt_focal/best", help="Dir del 1er modelo (best)")
    ap.add_argument("--modelo2", default="", help="2do modelo -> activa ENSAMBLE (promedio de probs). Produccion: kie_layoutlmv3_focal/best")
    ap.add_argument("--peso", type=float, default=0.5, help="Peso de --modelo en el ensamble (0-1). Default 0.5")
    ap.add_argument("--salida", default="", help="Carpeta para escribir <id>.pred.json (opcional)")
    ap.add_argument("--sin_regex", action="store_true", help="No limpiar con regex (texto crudo del span)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensamble = bool(args.modelo2)

    if ensamble:
        print(f"device: {device} | ENSAMBLE FOCAL: {args.modelo} (w={args.peso}) + {args.modelo2} (w={round(1-args.peso,3)})")
        proc1, model1, id2label1, ui1 = cargar_modelo(args.modelo, device)
        proc2, model2, _,         ui2 = cargar_modelo(args.modelo2, device)
        LABELS = [id2label1[i] for i in range(len(id2label1))]      # orden comun (del modelo 1)

        def predictor(words, boxes, imagen_path):
            P1 = predecir_probs(words, boxes, proc1, model1, ui1, device, LABELS, imagen_path)
            P2 = predecir_probs(words, boxes, proc2, model2, ui2, device, LABELS, imagen_path)
            Pe = args.peso * P1 + (1 - args.peso) * P2
            return [LABELS[i] for i in Pe.argmax(1)]
    else:
        print("device:", device, "| modelo unico:", args.modelo)
        proc, model, id2label, usa_imagen = cargar_modelo(args.modelo, device)

        def predictor(words, boxes, imagen_path):
            return predecir_labels(words, boxes, proc, model, id2label, usa_imagen, device,
                                   imagen_path=imagen_path)

    if args.ocr_json:
        jsons = [args.ocr_json]
    else:
        jsons = sorted(f for f in glob.glob(os.path.join(args.ocr_dir, "*.json"))
                       if not os.path.basename(f).startswith("_"))
    print(f"facturas a procesar: {len(jsons)}")
    if args.salida:
        os.makedirs(args.salida, exist_ok=True)

    for jp in jsons:
        res = procesar_json(jp, predictor, limpiar_regex=not args.sin_regex)
        print("\n" + "=" * 70)
        print(res["id"], "| CUFE:", (res.get("cufe") or "—"))
        for k, v in res.get("campos", {}).items():
            print(f"  {k:<28} {v}")
        if res.get("error"):
            print("  ERROR:", res["error"])
        if args.salida:
            with open(os.path.join(args.salida, res["id"] + ".pred.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
