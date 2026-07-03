#!/usr/bin/env python3
"""
pre_etiquetar_facturas.py
Capa 2 del pipeline de facturas: lee los JSON producidos por extraer_facturas.py
(tokens con label="O") y les asigna etiquetas BIO usando Qwen2.5:7b via Ollama.

Campos KIE (8):
  FACTURA    : NUMERO_FACTURA, FECHA_GENERACION
  EMISOR     : NIT_EMISOR, RAZON_SOCIAL_EMISOR
  ADQUIRIENTE: NIT_ADQUIRIENTE, RAZON_SOCIAL_ADQUIRIENTE
  TOTALES    : VALOR_TOTAL, IVA_TOTAL
  (CUFE/QR se extraen sin OCR en extraer_facturas.py — no necesitan BIO label)

Salida:
  - JSON actualizado por factura (con labels BIO en cada token)
  - label_studio_import.json  → importar directamente en Label Studio

Uso (VPS con GPU, venv activado):
  python pre_etiquetar_facturas.py \
      --ocr_dir  ~/agente-documentos/facturas_ocr \
      --salida_dir ~/agente-documentos/facturas_etiquetadas \
      --ls_output  ~/agente-documentos/label_studio_import.json
"""

import json
import os
import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional

import requests

# ── Configuracion ──────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_LLM = "qwen2.5:7b"

PROMPT = """\
Eres un extractor de datos de facturas electronicas colombianas (DIAN).
Analiza el texto OCR y extrae los siguientes campos:

FACTURA:
- NUMERO_FACTURA: numero o consecutivo de la factura. Puede tener prefijo de letras \
(ej: FE-1234, SETP990000001, FEV 12345). Aparece cerca de "Factura de Venta No." o similar.
- FECHA_GENERACION: fecha de generacion/emision impresa en la factura (NO la de vencimiento). \
Copiala EXACTAMENTE como aparece en el texto (mismo formato: AAAA-MM-DD, DD/MM/AAAA, etc.).

EMISOR (quien expide la factura, aparece en la parte SUPERIOR de la factura):
- NIT_EMISOR: NIT del EMISOR. Solo digitos, puede incluir DV (ej: 901234567-1 o 901234567).
- RAZON_SOCIAL_EMISOR: nombre o razon social del EMISOR exactamente como aparece.

ADQUIRIENTE (quien compra/recibe la factura, aparece en seccion separada):
- NIT_ADQUIRIENTE: NIT o cedula del ADQUIRIENTE/CLIENTE.
- RAZON_SOCIAL_ADQUIRIENTE: nombre o razon social del ADQUIRIENTE.

TOTALES:
- VALOR_TOTAL: valor TOTAL a pagar de la factura (el total final de la seccion de TOTALES). \
NO el valor/precio de un producto o linea individual. Solo el numero con puntos de miles \
(ej: 1.190.000), sin signo de moneda ni letras.
- IVA_TOTAL: valor TOTAL del IVA de la factura (el IVA consolidado de la seccion de TOTALES, junto \
al valor total). NO el IVA de un producto o linea individual. Solo el numero (ej: 19.000), sin \
signo ni letras. Si la factura no discrimina IVA (o es 0), usa null.

Responde SOLO con JSON valido, sin explicaciones, sin markdown:
{{"NUMERO_FACTURA": "...", "FECHA_GENERACION": "...", "NIT_EMISOR": "...", \
"RAZON_SOCIAL_EMISOR": "...", "NIT_ADQUIRIENTE": "...", "RAZON_SOCIAL_ADQUIRIENTE": "...", \
"VALOR_TOTAL": "...", "IVA_TOTAL": "..."}}

Si no encuentras un campo usa null.

Texto OCR de la factura:
{texto}"""

# ── Utilidades de normalizacion ────────────────────────────────────────────────
def _norm(text: str) -> str:
    """Minusculas, sin tildes, solo alfanumericos y espacios."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def _digitos(text: str) -> str:
    return re.sub(r"\D", "", text or "")

# ── LLM ───────────────────────────────────────────────────────────────────────
def extraer_con_llm(tokens: list[dict]) -> dict:
    texto_ocr = " ".join(t["text"] for t in tokens)
    payload = {
        "model":   MODELO_LLM,
        "prompt":  PROMPT.format(texto=texto_ocr[:5000]),
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.0, "num_predict": 600},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return json.loads(resp.json().get("response", "{}"))
    except Exception as e:
        print(f"    [LLM ERROR] {e}")
        return {}

# ── Matching valor LLM → indices de tokens ────────────────────────────────────
def _match_nit(tokens: list[dict], valor: Optional[str]) -> list[int]:
    """Match por digitos: NIT es 8-10 digitos, puede tener DV separado."""
    dv = _digitos(valor)
    if len(dv) < 7:
        return []
    for i, t in enumerate(tokens):
        dt = _digitos(t["text"])
        if dt == dv or dt == dv[:-1] or dv == dt[:-1]:
            return [i]
        if len(dt) >= 7 and dt in dv:
            return [i]
    return []

def _match_valor_total(tokens: list[dict], valor: Optional[str]) -> list[int]:
    """Match por digitos: el total es el numero con esos mismos digitos."""
    dv = _digitos(valor)
    if len(dv) < 3:
        return []
    for i, t in enumerate(tokens):
        if _digitos(t["text"]) == dv:
            return [i]
    return []

def _match_fecha(tokens: list[dict], valor: Optional[str]) -> list[int]:
    """Fecha de generacion: por digitos; tolera distinto orden (DD/MM/AAAA vs AAAA-MM-DD)."""
    dv = _digitos(valor)
    if len(dv) < 6:
        return []
    sdv = sorted(dv)
    # 1) token con los mismos digitos exactos
    for i, t in enumerate(tokens):
        if _digitos(t["text"]) == dv:
            return [i]
    # 2) mismo multiset de digitos (mismo dia/mes/anio en otro orden)
    for i, t in enumerate(tokens):
        td = _digitos(t["text"])
        if len(td) == len(dv) and sorted(td) == sdv:
            return [i]
    # 3) ensamblar tokens contiguos (dd, mm, aaaa partidos por OCR)
    for i in range(len(tokens)):
        acc = ""
        for j in range(i, min(i + 4, len(tokens))):
            acc += _digitos(tokens[j]["text"])
            if acc == dv or (len(acc) == len(dv) and sorted(acc) == sdv):
                return list(range(i, j + 1))
    return []

def _alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

def _match_numero_factura(tokens: list[dict], valor: Optional[str]) -> list[int]:
    """Numero de factura: alfanumerico, puede tener prefijo de letras y venir partido por OCR."""
    v = _alnum(valor)
    if len(v) < 3:
        return []
    norms = [_alnum(t["text"]) for t in tokens]
    # 1) token unico exacto
    for i, tn in enumerate(norms):
        if tn and tn == v:
            return [i]
    # 2) ensamblar tokens contiguos (prefijo + numero partidos por OCR)
    for i in range(len(tokens)):
        acc = ""
        for j in range(i, min(i + 4, len(tokens))):
            acc += norms[j]
            if acc == v:
                return list(range(i, j + 1))
    # 3) token que contiene el valor (o viceversa), con guarda de longitud
    for i, tn in enumerate(norms):
        if len(tn) >= 4 and (v in tn or (len(v) >= 5 and tn in v)):
            return [i]
    return []

def _match_multitoken(tokens: list[dict], valor: Optional[str], max_gap: int = 20) -> list[int]:
    """Sliding window con Jaccard normalizado."""
    if not valor:
        return []
    valor_n = _norm(valor)
    set_v = set(valor_n.split())
    if not set_v:
        return []

    tokens_n = [_norm(t["text"]) for t in tokens]
    mejor = (0.0, -1, -1)

    for i in range(len(tokens)):
        acum = ""
        for j in range(i, min(i + max_gap, len(tokens))):
            acum = (acum + " " + tokens_n[j]).strip()
            set_w = set(acum.split())
            union = set_v | set_w
            if not union:
                continue
            jaccard = len(set_v & set_w) / len(union)
            if jaccard > mejor[0]:
                mejor = (jaccard, i, j + 1)

    if mejor[0] >= 0.50:
        return list(range(mejor[1], mejor[2]))
    return []

def asignar_bio(tokens: list[dict], indices: list[int], entidad: str):
    for k, idx in enumerate(indices):
        tokens[idx]["label"] = f"B-{entidad}" if k == 0 else f"I-{entidad}"

# ── Pre-etiquetado de una factura ──────────────────────────────────────────────
MATCHERS = [
    ("NIT_EMISOR",              _match_nit),
    ("NIT_ADQUIRIENTE",         _match_nit),
    ("VALOR_TOTAL",             _match_valor_total),
    ("IVA_TOTAL",               _match_valor_total),
    ("NUMERO_FACTURA",          _match_numero_factura),
    ("FECHA_GENERACION",        _match_fecha),
    ("RAZON_SOCIAL_EMISOR",     _match_multitoken),
    ("RAZON_SOCIAL_ADQUIRIENTE",_match_multitoken),
]

def pre_etiquetar(factura: dict) -> dict[str, bool]:
    """
    Modifica factura['tokens'] in-place asignando labels BIO.
    Retorna dict entidad->bool indicando si se encontro cada campo.
    """
    tokens = factura["tokens"]

    for t in tokens:
        t["label"] = "O"

    campos = extraer_con_llm(tokens)
    factura["campos_llm"] = campos

    stats = {}
    for entidad, fn in MATCHERS:
        valor = campos.get(entidad)
        indices = fn(tokens, valor) if valor else []
        if indices:
            asignar_bio(tokens, indices, entidad)
        stats[entidad] = bool(indices)
        estado = f"OK ({len(indices)} token{'s' if len(indices)>1 else ''})" if indices else "NO ENCONTRADO"
        print(f"    {entidad:<28} '{str(valor)[:40]}'  →  {estado}")

    return stats

# ── Export Label Studio ────────────────────────────────────────────────────────
def factura_a_label_studio(factura: dict) -> dict:
    """
    Convierte un JSON de factura al formato de pre-anotacion de Label Studio.
    Las coordenadas se expresan en porcentajes del tamanio de la imagen.
    """
    w = factura.get("img_width", 1)
    h = factura.get("img_height", 1)
    resultado = []

    for i, token in enumerate(factura["tokens"]):
        if token["label"] == "O":
            continue
        x1, y1, x2, y2 = token["bbox"]
        entidad = token["label"].split("-", 1)[1]
        resultado.append({
            "id":              f"t{i}",
            "type":            "rectanglelabels",
            "from_name":       "label",
            "to_name":         "image",
            "original_width":  w,
            "original_height": h,
            "value": {
                "x":      round(x1 / w * 100, 2),
                "y":      round(y1 / h * 100, 2),
                "width":  round((x2 - x1) / w * 100, 2),
                "height": round((y2 - y1) / h * 100, 2),
                "rotation": 0,
                "rectanglelabels": [entidad],
            },
        })

    img_path = factura.get("imagen", "")
    ls_image = f"/data/local-files/?d={img_path}" if img_path.startswith("/") else img_path

    return {
        "data": {"image": ls_image},
        "predictions": [{
            "model_version": "heuristica_llm_v1",
            "score": 0.85,
            "result": resultado,
        }],
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Pre-etiquetado KIE con Qwen2.5:7b")
    ap.add_argument("--ocr_dir",    required=True, help="Dir con JSONs de extraer_facturas.py")
    ap.add_argument("--salida_dir", required=True, help="Dir de salida con JSONs etiquetados")
    ap.add_argument("--ls_output",  required=True, help="Archivo JSON para importar en Label Studio")
    ap.add_argument("--solo_nuevas", action="store_true",
                    help="Pre-etiqueta solo los JSONs sin salida previa (el ls_output trae solo las nuevas)")
    args = ap.parse_args()

    ocr_dir    = Path(args.ocr_dir)
    salida_dir = Path(args.salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(f for f in ocr_dir.glob("*.json") if not f.name.startswith("_"))
    if not jsons:
        print(f"No hay JSONs en {ocr_dir}")
        sys.exit(1)

    if args.solo_nuevas:
        ya = {p.name for p in salida_dir.glob("*.json")}
        pendientes = [f for f in jsons if f.name not in ya]
        print(f"Modo --solo_nuevas: {len(ya)} ya etiquetadas, {len(pendientes)} nuevas por etiquetar")
        if not pendientes:
            print("No hay facturas nuevas que etiquetar. Nada que hacer.")
            sys.exit(0)
        jsons = pendientes

    ls_items = []
    totales  = {e: 0 for e, _ in MATCHERS}
    errores  = 0

    for i, json_path in enumerate(jsons, 1):
        print(f"\n[{i}/{len(jsons)}] {json_path.name}")
        try:
            factura = json.loads(json_path.read_text(encoding="utf-8"))
            stats   = pre_etiquetar(factura)

            out = salida_dir / json_path.name
            out.write_text(json.dumps(factura, ensure_ascii=False, indent=2), encoding="utf-8")

            ls_items.append(factura_a_label_studio(factura))

            for e, ok in stats.items():
                if ok:
                    totales[e] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errores += 1

    ls_path = Path(args.ls_output)
    ls_path.write_text(json.dumps(ls_items, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(jsons) - errores
    print(f"\n{'=' * 60}")
    print(f"Procesadas: {n}/{len(jsons)} ({errores} errores)")
    print(f"\n  {'Campo':<28} {'Encontrado':>12}   {'%':>5}")
    print(f"  {'-'*50}")
    for entidad, _ in MATCHERS:
        cnt = totales[entidad]
        pct = cnt / n * 100 if n else 0
        barra = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {entidad:<28} {cnt:>5}/{n:<5}  {barra}  {pct:.0f}%")

    print(f"\nJSONs etiquetados : {salida_dir}/")
    print(f"Label Studio file : {ls_path}")

    print(f"""
Siguiente: instalar y arrancar Label Studio en el VPS
  pip install label-studio
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/ \\
  label-studio start --port 8080

Luego hacer tunel SSH desde Windows:
  gcloud compute ssh agente-vps --zone=us-west1-a -- -L 8080:localhost:8080
  Abrir: http://localhost:8080
  Importar: {ls_path}
""")


if __name__ == "__main__":
    main()
