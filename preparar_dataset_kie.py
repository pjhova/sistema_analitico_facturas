#!/usr/bin/env python3
"""
preparar_dataset_kie.py
Capa 3 del pipeline de facturas: convierte (tokens OCR + anotacion de Label Studio)
en el dataset token-level (BIO) que consumen LayoutLMv3 / LiLT, con split por layout.

NO re-OCR ni re-anota: REUSA los tokens de extraer_facturas.py (facturas_ocr/*.json)
y proyecta sobre ellos las cajas de tu JSON de Label Studio (por solapamiento/IoU).

Entradas:
  - --ocr_dir   : carpeta con los JSON de extraer_facturas.py (tokens: text, bbox px, img_width/height)
  - --ls_json   : tu archivo de Label Studio (import con 'predictions' o export con 'annotations')

Salidas (en --salida_dir):
  - train.jsonl / test.jsonl : un ejemplo por factura
        {id, image_path, layout, tokens[str], bboxes[[x,y,x,y] 0-1000], ner_tags[str]}
  - labels.txt               : lista de etiquetas (O primero, luego B-/I- por entidad)
  - _resumen_dataset.txt     : conteos y advertencias

Uso:
  python preparar_dataset_kie.py \
      --ocr_dir   ~/agente-documentos/facturas_ocr \
      --ls_json   ~/agente-documentos/label_studio_import.json \
      --salida_dir ~/agente-documentos/dataset_kie \
      --n_test_layouts 6
"""
import os, re, json, glob, argparse, random, statistics
from collections import defaultdict

# ── Parseo del JSON de Label Studio ─────────────────────────────────────────────
def _stem_from_ls_image(url: str) -> str:
    """De 'http://localhost:8081/NAME.png' o '/data/local-files/?d=/ruta/NAME.png' -> 'NAME'."""
    if "d=" in url:
        url = url.split("d=")[-1]
    name = os.path.basename(url.split("?")[0])
    return os.path.splitext(name)[0]

def _result_de_task(task: dict) -> list:
    """Toma el result de annotations (gold) si existe; si no, de predictions (pre-anotaciones)."""
    for key in ("annotations", "predictions"):
        for a in (task.get(key) or []):
            res = a.get("result") or []
            if res:
                return res
    return []

def _boxes_de_result(result: list) -> list:
    """Extrae cajas {entity, x, y, width, height} (en % de la imagen) del result de Label Studio."""
    boxes = []
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        v = r.get("value", {})
        labs = v.get("rectanglelabels") or []
        if not labs:
            continue
        boxes.append({"entity": labs[0], "x": v["x"], "y": v["y"],
                      "width": v["width"], "height": v["height"]})
    return boxes

def cargar_ls(ls_json: str) -> dict:
    """stem de imagen -> lista de cajas. Acepta un array de tasks o un solo task."""
    data = json.loads(open(ls_json, encoding="utf-8").read())
    if isinstance(data, dict):
        data = [data]
    por_stem = {}
    for task in data:
        img = (task.get("data") or {}).get("image", "")
        if not img:
            continue
        stem = _stem_from_ls_image(img)
        por_stem[stem] = _boxes_de_result(_result_de_task(task))
    return por_stem

# ── Geometria ───────────────────────────────────────────────────────────────────
def _ls_a_pixeles(b: dict, W: int, H: int) -> list:
    """Caja de Label Studio (% ) -> pixeles [x1,y1,x2,y2] en el espacio de la imagen."""
    x1 = b["x"] / 100.0 * W
    y1 = b["y"] / 100.0 * H
    x2 = (b["x"] + b["width"]) / 100.0 * W
    y2 = (b["y"] + b["height"]) / 100.0 * H
    return [x1, y1, x2, y2]

def _containment(a: list, b: list) -> float:
    """Fraccion del area de 'a' (token) contenida en 'b' (caja de campo)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a

def _norm_bbox(box: list, W: int, H: int) -> list:
    """Normaliza a la escala 0-1000 (requisito de LayoutLMv3/LiLT), con clamp."""
    def c(v, lim):
        return min(1000, max(0, int(round(v / lim * 1000))))
    return [c(box[0], W), c(box[1], H), c(box[2], W), c(box[3], H)]

def _cx(b): return (b[0] + b[2]) / 2.0
def _cy(b): return (b[1] + b[3]) / 2.0

# ── Asignacion de etiquetas BIO ─────────────────────────────────────────────────
def asignar_bio(tokens: list, ls_boxes: list, W: int, H: int, thr: float = 0.5):
    """
    Devuelve (labels, no_emparejadas):
      labels: lista BIO alineada a tokens.
      no_emparejadas: entidades cuyo box no recibio ningun token (campo perdido).
    """
    n = len(tokens)
    labels = ["O"] * n
    nb = len(ls_boxes)
    # tolerancia de fila para ordenar lectura (segun altura mediana del token)
    alturas = [max(1, t["bbox"][3] - t["bbox"][1]) for t in tokens] or [12]
    row_tol = max(8.0, statistics.median(alturas) * 0.6)

    cajas_px = [_ls_a_pixeles(b, W, H) for b in ls_boxes]

    # Matriz de solapamiento SIMETRICO token x caja:
    #   max(fraccion del token dentro de la caja, fraccion de la caja cubierta por
    #   el token). El 2do termino captura el caso del OCR que funde rotulo+valor en
    #   un token mucho mas ancho que la caja (ej. 'NIT: 900.319.753-3'): token-en-caja
    #   es bajo, pero la caja queda casi entera dentro del token. Los campos
    #   multi-token (razon social en varias palabras) siguen casando por el 1er termino.
    ov = [[0.0] * nb for _ in range(n)]
    for ti, t in enumerate(tokens):
        for bi, fb in enumerate(cajas_px):
            ov[ti][bi] = max(_containment(t["bbox"], fb), _containment(fb, t["bbox"]))

    # 1) cada token -> su mejor caja (si supera el umbral)
    asign = [-1] * n  # ti -> bi
    grupos = defaultdict(list)
    for ti in range(n):
        if nb == 0:
            break
        bi = max(range(nb), key=lambda b: ov[ti][b])
        if ov[ti][bi] >= thr:
            asign[ti] = bi
            grupos[bi].append(ti)

    # 2) PASADA DE RESCATE de cajas desabastecidas.
    #    Efecto secundario del solapamiento simetrico: un token de VALOR puro
    #    (ej. 'NIT: 890900307-7') cae dentro de la caja grande vecina (RAZON_SOCIAL,
    #    caja-en-token=1.0) y se la lleva esa, dejando huerfana la caja del NIT (y
    #    mal-etiquetado el NIT como razon social). Una caja sin tokens recupera su
    #    token de mayor solapamiento SOLO si la caja donante conserva >=1 token: asi
    #    se rescata el NIT y se corrige su etiqueta, sin tocar los campos multi-token
    #    ni los tokens GENUINAMENTE fundidos (un unico token con ambos campos, ej.
    #    'FARMATODO COLOMBIA S.A NIT: 830129327-1'): ahi la donante tiene 1 solo token
    #    -> no se roba -> se delega a la capa de regex post-KIE.
    for bi in range(nb):
        if grupos[bi]:
            continue
        cands = sorted(((ov[ti][bi], ti) for ti in range(n) if ov[ti][bi] >= thr), reverse=True)
        for _, ti in cands:
            donante = asign[ti]
            if donante >= 0 and len(grupos[donante]) > 1:
                grupos[donante].remove(ti)
                asign[ti] = bi
                grupos[bi].append(ti)
                break

    # 3) BIO por caja, en orden de lectura
    for bi, idxs in grupos.items():
        if not idxs:
            continue
        ent = ls_boxes[bi]["entity"]
        idxs.sort(key=lambda i: (round(_cy(tokens[i]["bbox"]) / row_tol), _cx(tokens[i]["bbox"])))
        for k, ti in enumerate(idxs):
            labels[ti] = ("B-" if k == 0 else "I-") + ent

    no_emparejadas = [ls_boxes[bi]["entity"] for bi in range(nb) if not grupos[bi]]
    return labels, no_emparejadas

# ── Layout ───────────────────────────────────────────────────────────────────────
def layout_de(stem: str, regex: str) -> str:
    m = re.match(regex, stem)
    return m.group(1) if m else stem

# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Construye el dataset KIE (token-level BIO) con split por layout")
    ap.add_argument("--ocr_dir",   required=True, help="Dir con JSON de extraer_facturas.py")
    ap.add_argument("--ls_json",   required=True, help="JSON de Label Studio (import o export)")
    ap.add_argument("--salida_dir", required=True, help="Dir de salida del dataset")
    ap.add_argument("--layout_regex", default=r"(layout_\d+)_",
                    help=r"Regex; group(1)=layout=emisor. Default: '(layout_\d+)_' agrupa 1 pag "
                         r"(layout_08_014) y multipag (layout_09_014_p1/_p2) bajo el mismo emisor (layout_09)")
    ap.add_argument("--n_test_layouts", type=int, default=6, help="N layouts para test (split por layout)")
    ap.add_argument("--test_layouts", default="", help="Lista explicita de layouts de test (coma-separados)")
    ap.add_argument("--containment", type=float, default=0.5,
                    help="Umbral de solapamiento SIMETRICO max(token-en-caja, caja-en-token). "
                         "Con el criterio simetrico, 0.5 ya captura tokens fundidos rotulo+valor.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.salida_dir, exist_ok=True)
    ls_por_stem = cargar_ls(args.ls_json)
    print(f"Label Studio: {len(ls_por_stem)} facturas anotadas")

    ocr_jsons = sorted(f for f in glob.glob(os.path.join(args.ocr_dir, "*.json"))
                       if not os.path.basename(f).startswith("_"))

    ejemplos = []          # dicts del dataset
    entidades = set()
    avisos = []
    sin_ls = 0
    for jpath in ocr_jsons:
        stem = os.path.splitext(os.path.basename(jpath))[0]
        if stem not in ls_por_stem:
            sin_ls += 1
            continue
        fac = json.loads(open(jpath, encoding="utf-8").read())
        tokens = fac.get("tokens", [])
        W = fac.get("img_width"); H = fac.get("img_height")
        if not tokens or not W or not H:
            avisos.append(f"{stem}: sin tokens o sin img_width/height -> omitido")
            continue

        ls_boxes = ls_por_stem[stem]
        labels, no_emp = asignar_bio(tokens, ls_boxes, W, H, thr=args.containment)
        for e in no_emp:
            avisos.append(f"{stem}: la caja '{e}' no caso con ningun token OCR (campo perdido)")
        for b in ls_boxes:
            entidades.add(b["entity"])

        ejemplos.append({
            "id": stem,
            "image_path": fac.get("imagen", ""),
            "layout": layout_de(stem, args.layout_regex),
            "tokens": [t["text"] for t in tokens],
            "bboxes": [_norm_bbox(t["bbox"], W, H) for t in tokens],
            "ner_tags": labels,
        })

    if not ejemplos:
        print("No se generaron ejemplos (¿coinciden los nombres entre OCR y Label Studio?).")
        return

    # ── Etiquetas (O primero, luego B-/I- por entidad ordenada) ──────────────────
    labels_list = ["O"]
    for e in sorted(entidades):
        labels_list += [f"B-{e}", f"I-{e}"]
    with open(os.path.join(args.salida_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels_list) + "\n")

    # ── Split por layout ─────────────────────────────────────────────────────────
    layouts = sorted({e["layout"] for e in ejemplos})
    if args.test_layouts.strip():
        test_set = {s.strip() for s in args.test_layouts.split(",") if s.strip()}
    else:
        rnd = random.Random(args.seed)
        k = min(args.n_test_layouts, max(1, len(layouts) - 1))
        test_set = set(rnd.sample(layouts, k))

    train = [e for e in ejemplos if e["layout"] not in test_set]
    test  = [e for e in ejemplos if e["layout"] in test_set]

    def escribir(path, filas):
        with open(path, "w", encoding="utf-8") as f:
            for e in filas:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    escribir(os.path.join(args.salida_dir, "train.jsonl"), train)
    escribir(os.path.join(args.salida_dir, "test.jsonl"), test)

    # ── Resumen ──────────────────────────────────────────────────────────────────
    dist = defaultdict(int)
    for e in ejemplos:
        for lab in e["ner_tags"]:
            dist[lab] += 1
    lineas = []
    lineas.append(f"Facturas con dataset: {len(ejemplos)}  (sin anotacion LS y omitidas: {sin_ls})")
    lineas.append(f"Layouts: {len(layouts)}  |  test layouts ({len(test_set)}): {sorted(test_set)}")
    lineas.append(f"Split -> train: {len(train)} facturas | test: {len(test)} facturas")
    lineas.append(f"Etiquetas ({len(labels_list)}): {labels_list}")
    lineas.append("\nDistribucion de etiquetas (conteo de tokens):")
    for lab in labels_list:
        lineas.append(f"  {lab:<28} {dist.get(lab,0)}")
    if avisos:
        lineas.append(f"\nAVISOS ({len(avisos)}):")
        lineas += [f"  - {a}" for a in avisos]
    resumen = "\n".join(lineas)
    with open(os.path.join(args.salida_dir, "_resumen_dataset.txt"), "w", encoding="utf-8") as f:
        f.write(resumen + "\n")

    print(resumen)
    print(f"\nDataset escrito en: {args.salida_dir}/  (train.jsonl, test.jsonl, labels.txt)")
    print("Siguiente: entrenar_kie.py (fine-tune LayoutLMv3 vs LiLT) sobre este dataset.")

if __name__ == "__main__":
    main()
