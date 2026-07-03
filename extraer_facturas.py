#!/usr/bin/env python3
"""
Extraccion base de facturas electronicas (proyecto facturas DIAN).

Por cada imagen de factura produce un JSON con:
  - id      : contador asignado
  - cufe    : decodificado del QR (codigo completo, no por OCR)
  - qr_raw  : contenido crudo del QR (para verificar)
  - tokens  : lista {text, bbox, score, label} del OCR (label="O" por defecto,
              listos para ANOTAR los 8 campos KIE: NUMERO_FACTURA, FECHA_GENERACION,
              NIT_EMISOR, RAZON_SOCIAL_EMISOR, NIT_ADQUIRIENTE, RAZON_SOCIAL_ADQUIRIENTE,
              VALOR_TOTAL, IVA_TOTAL)

Tambien escribe un CSV resumen: id, archivo, cufe, n_tokens.

NO etiqueta los campos KIE (eso lo hace el modelo despues de anotar). FECHA fue eliminada
del alcance; ID, CUFE y QR son automaticos (sin OCR).

Uso (en la VM con GPU):
  python extraer_facturas.py --facturas_dir facturas --salida_dir facturas_ocr --device gpu
"""
import os, re, csv, json, glob, argparse
from pathlib import Path

import cv2  # QR
import fitz  # PyMuPDF: conversion PDF -> PNG
from PIL import Image  # tamano de imagen
from paddleocr import PaddleOCR

# --------------------------------------------------------------------------------------
# Conversion PDF -> PNG (facturas que llegan en PDF)
# --------------------------------------------------------------------------------------
def pdf_a_png(pdf_path, salida_png_dir, dpi=200):
    """Renderiza cada pagina del PDF a un PNG. Devuelve lista de rutas PNG generadas."""
    os.makedirs(salida_png_dir, exist_ok=True)
    rutas = []
    stem = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0  # 72 = DPI base de PDF
    mat = fitz.Matrix(zoom, zoom)
    for n, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        # una sola pagina -> sin sufijo; multipagina -> _pN
        nombre = f"{stem}.png" if doc.page_count == 1 else f"{stem}_p{n+1}.png"
        out = os.path.join(salida_png_dir, nombre)
        pix.save(out)
        rutas.append(out)
    doc.close()
    return rutas

# --------------------------------------------------------------------------------------
# CUFE desde el QR
# --------------------------------------------------------------------------------------
_detector_qr = cv2.QRCodeDetector()

# pyzbar es mucho mas robusto que cv2.QRCodeDetector (opcional pero recomendado)
try:
    from pyzbar.pyzbar import decode as _zbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False

def _qr_con_pyzbar(img_bgr):
    """Intenta pyzbar sobre la imagen original y una version 2x (QR pequenos)."""
    h, w = img_bgr.shape[:2]
    variantes = [img_bgr]
    variantes.append(cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    for im in variantes:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        try:
            for obj in _zbar_decode(gray):
                data = obj.data.decode("utf-8", errors="ignore")
                if data:
                    return data
        except Exception:
            pass
    return ""

def leer_qr(img_path):
    """Decodifica el/los QR de la imagen. Devuelve el texto crudo del QR (o '')."""
    img = cv2.imread(img_path)
    if img is None:
        return ""
    # 1) pyzbar (preferido: detecta QR pequenos/de baja calidad)
    if _HAS_PYZBAR:
        data = _qr_con_pyzbar(img)
        if data:
            return data
    # 2) cv2: intento simple
    data, pts, _ = _detector_qr.detectAndDecode(img)
    if data:
        return data
    # 3) cv2: intento multiple (algunas facturas tienen varios codigos)
    try:
        ok, datos, pts, _ = _detector_qr.detectAndDecodeMulti(img)
        if ok and datos:
            for d in datos:
                if d:
                    return d
    except Exception:
        pass
    return ""

def extraer_cufe(qr_text):
    """Saca el CUFE del contenido del QR (URL DIAN con documentkey, o codigo largo)."""
    if not qr_text:
        return None
    # 1) URL DIAN con documentkey=<CUFE>
    m = re.search(r"documentkey=([0-9A-Za-z]+)", qr_text)
    if m:
        return m.group(1)
    # 2) algun campo CUFE=... o cadena larga alfanumerica
    m = re.search(r"\b([0-9A-Fa-f]{40,})\b", qr_text)
    if m:
        return m.group(1)
    return qr_text.strip()

# --------------------------------------------------------------------------------------
# OCR (tokens para anotar KIE)
# --------------------------------------------------------------------------------------
def crear_ocr(device="cpu"):
    # Flags criticos del proyecto (ver CLAUDE.md):
    #  - use_doc_unwarping=False -> NO cargar UVDoc (deforma coordenadas, bug #1)
    #  - desactivar orientacion (PDFs/facturas planos)
    #  - enable_mkldnn=False -> evita el crash oneDNN PIR en CPU
    return PaddleOCR(
        lang="es",
        device=device,
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

def _poly_a_box(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]

def ocr_tokens(ocr, img_path):
    res = ocr.predict(img_path)
    tokens = []
    for r in res:
        if hasattr(r, "get"):
            textos = r.get("rec_texts", []); scores = r.get("rec_scores", [])
            polys  = r.get("rec_polys", []) or r.get("dt_polys", [])
        else:
            textos = getattr(r, "rec_texts", []); scores = getattr(r, "rec_scores", [])
            polys  = getattr(r, "rec_polys", None) or getattr(r, "dt_polys", [])
        for t, s, p in zip(textos, scores, polys):
            if not t.strip():
                continue
            tokens.append({
                "text": t.strip(),
                "bbox": _poly_a_box(p),
                "score": float(s),
                "label": "O",   # por defecto; se anota despues (B-NIT, B-VALOR_TOTAL, ...)
            })
    return tokens

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
EXTS_PDF = ("*.pdf", "*.PDF")

def main():
    ap = argparse.ArgumentParser(description="Extraccion base de facturas (FECHA+CUFE+OCR)")
    ap.add_argument("--facturas_dir", required=True, help="Carpeta con facturas (imagenes o PDF)")
    ap.add_argument("--salida_dir", default="facturas_ocr")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--solo_nuevas", action="store_true",
                    help="Procesa solo facturas sin JSON de salida previo (no re-procesa las ya hechas)")
    args = ap.parse_args()

    os.makedirs(args.salida_dir, exist_ok=True)
    png_dir = os.path.join(args.salida_dir, "_png")  # PNGs renderizados de PDFs

    # Stems ya procesados (para modo --solo_nuevas)
    existentes = set()
    if args.solo_nuevas:
        existentes = {p.stem for p in Path(args.salida_dir).glob("*.json")
                      if not p.name.startswith("_")}

    # Descubrir archivos: imagenes directas + PDFs (se convierten a PNG)
    archivos_img = []
    for e in EXTS:
        archivos_img += glob.glob(os.path.join(args.facturas_dir, e))
    archivos_pdf = []
    for e in EXTS_PDF:
        archivos_pdf += glob.glob(os.path.join(args.facturas_dir, e))

    # Lista de trabajo: (png_path, source_path) — source es el PDF original o la misma imagen
    trabajo = []
    for p in sorted(archivos_img):
        if args.solo_nuevas and Path(p).stem in existentes:
            continue
        trabajo.append((p, p))
    for pdf in sorted(set(archivos_pdf)):
        stem = Path(pdf).stem
        # saltar PDFs ya procesados (incluye multipagina: stem o stem_pN)
        if args.solo_nuevas and any(s == stem or s.startswith(stem + "_p") for s in existentes):
            continue
        try:
            for png in pdf_a_png(pdf, png_dir):
                trabajo.append((png, pdf))
        except Exception as e:
            print(f"  ERROR convirtiendo {os.path.basename(pdf)}: {e}")

    if not trabajo:
        if args.solo_nuevas and existentes:
            print(f"No hay facturas NUEVAS. Ya hay {len(existentes)} procesadas en {args.salida_dir}/")
        else:
            print(f"No hay imagenes ni PDFs en {args.facturas_dir}")
        return
    if args.solo_nuevas:
        print(f"Modo --solo_nuevas: {len(existentes)} ya procesadas, {len(trabajo)} paginas NUEVAS a procesar")
    else:
        print(f"Encontradas {len(archivos_img)} imagenes + {len(archivos_pdf)} PDFs -> {len(trabajo)} paginas a procesar")

    print(f"Inicializando PaddleOCR (device={args.device})...")
    ocr = crear_ocr(args.device)

    resumen = []
    id_base = len(existentes)  # los nuevos continuan la numeracion
    for i, (img_path, source_path) in enumerate(trabajo, id_base + 1):
        stem = Path(img_path).stem
        try:
            qr_raw = leer_qr(img_path)
            cufe = extraer_cufe(qr_raw)
            tokens = ocr_tokens(ocr, img_path)
            with Image.open(img_path) as _img:
                img_w, img_h = _img.size

            registro = {
                "id": i,
                "imagen": os.path.abspath(img_path),
                "source": os.path.abspath(source_path),
                "img_width": img_w,
                "img_height": img_h,
                "cufe": cufe,
                "qr_raw": qr_raw,
                "tokens": tokens,
            }
            with open(os.path.join(args.salida_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(registro, f, ensure_ascii=False, indent=2)

            qr_ok = "QR_OK" if cufe else "SIN_QR"
            pos = i - id_base
            print(f"[{pos}/{len(trabajo)}] {stem}: {len(tokens)} tokens | {qr_ok}")
            resumen.append({"id": i, "archivo": os.path.basename(source_path),
                            "cufe": cufe or "", "n_tokens": len(tokens)})
        except Exception as e:
            pos = i - id_base
            print(f"[{pos}/{len(trabajo)}] {stem}: ERROR {e}")

    # CSV resumen (en modo --solo_nuevas se agrega al existente, no se sobrescribe)
    if resumen:
        csv_path = os.path.join(args.salida_dir, "_resumen.csv")
        append = args.solo_nuevas and os.path.exists(csv_path)
        with open(csv_path, "a" if append else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "archivo", "cufe", "n_tokens"])
            if not append:
                w.writeheader()
            w.writerows(resumen)

    sin_qr = sum(1 for r in resumen if not r["cufe"])
    print(f"\nListo. {len(resumen)} facturas | sin QR legible: {sin_qr}")
    print(f"Salida en: {args.salida_dir}/  (un JSON por factura + _resumen.csv)")
    print("PNGs renderizados de PDFs en: " + png_dir + "/")
    print("Siguiente: pre_etiquetar_facturas.py para asignar los 8 campos KIE.")

if __name__ == "__main__":
    main()
