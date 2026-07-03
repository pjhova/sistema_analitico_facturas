#!/usr/bin/env python3
"""
reglas_validacion.py
Validacion de facturas en DOS capas, sobre los campos extraidos por el KIE
(salida de inferir_factura.py):

  A) Reglas deterministas (sintacticas / de coherencia): DV del NIT (modulo 11),
     coherencia y tarifa de IVA, fecha valida, longitud del CUFE, obligatorios, formato.
  B) Cruce contra una base de datos de referencia (db_facturas.csv), POR CUFE:
     la validacion sigue la logica de la DIAN en dos pasos. (1) EXISTENCIA: se busca el
     CUFE extraido (del QR de la imagen) en la base; si esta, la factura existe (es real);
     si no, no existe / es falsa y no se comparan campos. (2) COHERENCIA: solo si el CUFE
     existe, se confronta cada campo extraido con el valor real de la BD. Esto detecta los
     valores "mal extraidos pero plausibles" que las reglas por si solas no atrapan. Las
     columnas que la BD tenga vacias quedan NO_COMPARABLE.
     La BD se lee con DuckDB (read_csv robusto: comillas, campos multilinea del QR,
     BOM y filas irregulares); el cruce/normalizacion fina (DV, montos CO/US, fechas,
     fuzzy de razon social) se hace en Python. Fallback a csv.reader con --motor csv.

NO se emite un veredicto global (VALIDA/NO_VALIDA): la salida es la VALIDEZ POR CAMPO.
Cada campo del cruce sale COINCIDE / DIFIERE / NO_COMPARABLE, y por factura se reporta
una validez en [0,1] = campos que COINCIDEN / campos comparables (p.ej. 0.8 = 4 de 5).
Las reglas deterministas se reportan aparte (PASA / OBSERVACION / FALLA / NO_APLICA).

Uso:
  python reglas_validacion.py --pred_dir pred_kie --db db_facturas.csv --salida validacion
  python reglas_validacion.py --pred_json pred_kie/layout_05_003.pred.json --db db_facturas.csv
  python reglas_validacion.py --pred_dir pred_kie            # solo reglas (sin cruce)
"""
import argparse, csv, glob, json, os, re, unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher

TARIFAS_IVA = (0.0, 0.05, 0.19)
PESOS_DV = [3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71]
# Llave de cruce con la BD: el CUFE. Una factura electronica con CUFE deberia existir en la
# base (que hace de base DIAN): si el CUFE esta -> existe; si no -> no existe / es falsa.
# La columna del CUFE en db_facturas.csv es CODIGO_CUFE. El CUFE de la factura sale del QR.
COL_CUFE_BD = "CODIGO_CUFE"
UMBRAL_RAZON = 0.80   # similitud minima para considerar coincidente una razon social

def _norm_cufe(s):
    """Normaliza un CUFE a 96 hex en minusculas (o hex-only si no hay match de 96)."""
    s = (s or "").lower()
    m = re.search(r"[0-9a-f]{96}", s)
    return m.group(0) if m else re.sub(r"[^0-9a-f]", "", s)


# ── normalizadores ───────────────────────────────────────────────────────────────
def _digitos(s):
    return re.sub(r"\D", "", s or "")

def _alnum(s):
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())

def _norm_texto(s):
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper()).strip()

def _parse_monto(s):
    """Parsea un monto en formato CO ('8.000,00') o US ('8,000.00') -> float."""
    s = re.sub(r"[^\d.,]", "", s or "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # coma es el decimal (CO)
            s = s.replace(".", "").replace(",", ".")
        else:                                     # punto es el decimal (US)
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    else:  # solo punto: si parece separador de miles, quitarlo
        if re.search(r"\.\d{3}(\D|$)", s) and not re.search(r"\.\d{1,2}$", s):
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None

def _parse_fecha(txt):
    """Fecha en varios formatos (incluye 'dd/mm/aaaa, hh:mm') -> date o None."""
    t = (txt or "").strip()
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", t)        # dd/mm/aaaa
    if m:
        try: return date(int(m[3]), int(m[2]), int(m[1]))
        except ValueError: pass
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", t)            # aaaa-mm-dd
    if m:
        try: return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError: pass
    if re.fullmatch(r"\d{8}", t):
        for fmt in ("%Y%m%d", "%d%m%Y"):
            try: return datetime.strptime(t, fmt).date()
            except ValueError: pass
    return None

def dv_nit(cuerpo):
    s = sum(int(d) * PESOS_DV[i] for i, d in enumerate(reversed(cuerpo)) if i < len(PESOS_DV))
    r = s % 11
    return r if r < 2 else 11 - r

def _cufe_de(cufe, qr_raw):
    for fuente in (cufe, qr_raw):
        if not fuente:
            continue
        m = re.search(r"[0-9a-fA-F]{96}", fuente)
        if m:
            return m.group(0).lower()
    return (cufe or "").strip().lower() or None


# ── comparadores campo a campo (extraido vs BD) ──────────────────────────────────
def _match_digitos(a, b):
    da, db = _digitos(a), _digitos(b)
    if not da or not db:
        return None                               # no comparable
    if da == db:
        return True
    # tolerancia de DV: uno es prefijo del otro y difieren en 1 digito
    corto, largo = sorted((da, db), key=len)
    return largo.startswith(corto) and len(largo) - len(corto) == 1

def _match_monto(a, b):
    va, vb = _parse_monto(a), _parse_monto(b)
    if va is None or vb is None:
        return None
    return abs(va - vb) <= max(1.0, 0.01 * max(va, vb))

def _match_fecha(a, b):
    fa, fb = _parse_fecha(a), _parse_fecha(b)
    if fa is None or fb is None:
        return None
    return fa == fb

def _match_alnum(a, b):
    aa, bb = _alnum(a), _alnum(b)
    if not aa or not bb:
        return None
    return aa == bb or aa in bb or bb in aa

def _match_razon(a, b):
    ta, tb = _norm_texto(a), _norm_texto(b)
    if not ta or not tb:
        return None
    return SequenceMatcher(None, ta, tb).ratio() >= UMBRAL_RAZON

COMPARADORES = {
    "FECHA_GENERACION": _match_fecha,
    "IVA_TOTAL": _match_monto,
    "VALOR_TOTAL": _match_monto,
    "NIT_EMISOR": _match_digitos,
    "NIT_ADQUIRIENTE": _match_digitos,
    "NUMERO_FACTURA": _match_alnum,
    "RAZON_SOCIAL_EMISOR": _match_razon,
    "RAZON_SOCIAL_ADQUIRIENTE": _match_razon,
}


# ── capa A: reglas deterministas ─────────────────────────────────────────────────
def validar_reglas(campos, cufe=None, qr_raw=None, hoy=None):
    hoy = hoy or date.today()
    R = []
    def add(n, nombre, estado, detalle, critica=False):
        R.append({"n": n, "regla": nombre, "estado": estado, "detalle": detalle, "critica": critica})
    g = lambda k: (campos.get(k) or "").strip()

    oblig = ["NUMERO_FACTURA", "FECHA_GENERACION", "NIT_EMISOR", "RAZON_SOCIAL_EMISOR",
             "NIT_ADQUIRIENTE", "VALOR_TOTAL"]
    faltan = [k for k in oblig if not g(k)]
    add(10, "Campos obligatorios presentes", "PASA" if not faltan else "FALLA",
        "todos presentes" if not faltan else f"faltan: {', '.join(faltan)}", critica=True)

    nit_e = _digitos(g("NIT_EMISOR"))
    if len(nit_e) >= 8:
        cuerpo, dv = nit_e[:-1], int(nit_e[-1])
        ok = dv_nit(cuerpo) == dv
        add(1, "DV del NIT emisor (modulo 11)", "PASA" if ok else "OBSERVACION",
            f"NIT {cuerpo}-{dv}; DV esperado {dv_nit(cuerpo)}" + ("" if ok else " (en datos anonimizados es esperable)"))
    else:
        add(1, "DV del NIT emisor (modulo 11)", "NO_APLICA", f"NIT con {len(nit_e)} digitos")

    nit_a = _digitos(g("NIT_ADQUIRIENTE"))
    ids_ok = (nit_e.isdigit() if nit_e else True) and (nit_a.isdigit() if nit_a else True)
    add(11, "Formato de identificadores (numericos)", "PASA" if ids_ok else "OBSERVACION",
        f"NIT_emisor='{g('NIT_EMISOR')}' NIT_adq='{g('NIT_ADQUIRIENTE')}'")

    f = _parse_fecha(g("FECHA_GENERACION"))
    if f is None:
        add(8, "Fecha valida", "OBSERVACION", f"no se pudo parsear '{g('FECHA_GENERACION')}'")
    elif f > hoy:
        add(8, "Fecha valida", "FALLA", f"fecha futura: {f.isoformat()}", critica=True)
    else:
        add(8, "Fecha valida", "PASA", f.isoformat())

    cu = _cufe_de(cufe, qr_raw)
    if not cu:
        add(5, "CUFE (96 hex)", "NO_APLICA", "sin CUFE/QR")
    elif re.fullmatch(r"[0-9a-f]{96}", cu):
        add(5, "CUFE (96 hex)", "PASA", f"{cu[:12]}… (len 96)")
    else:
        add(5, "CUFE (96 hex)", "OBSERVACION", f"longitud {len(cu)} o no-hex")

    total, iva = _digitos(g("VALOR_TOTAL")), _digitos(g("IVA_TOTAL"))
    if total and iva:
        ti, ii = int(total), int(iva)
        add(2, "Coherencia IVA <= total", "PASA" if ii <= ti else "FALLA",
            f"IVA {ii} {'<=' if ii <= ti else '>'} total {ti}", critica=ii > ti)
        base = ti - ii
        if base > 0:
            tarifa = ii / base
            cerca = min(TARIFAS_IVA, key=lambda t: abs(t - tarifa))
            add(4, "IVA por tarifa (0/5/19%)", "PASA" if abs(cerca - tarifa) <= 0.01 else "OBSERVACION",
                f"base {base}, tarifa {tarifa*100:.1f}% (cercana {cerca*100:.0f}%)")
        else:
            add(4, "IVA por tarifa (0/5/19%)", "NO_APLICA", "base no positiva")
    else:
        add(2, "Coherencia IVA <= total", "NO_APLICA", "falta VALOR_TOTAL o IVA_TOTAL")
        add(4, "IVA por tarifa (0/5/19%)", "NO_APLICA", "falta VALOR_TOTAL o IVA_TOTAL")

    for n, nombre, motivo in [
        (3, "Suma de lineas = subtotal", "no se extraen lineas"),
        (6, "Consecutivo en rango de resolucion", "no se extrae la resolucion"),
        (7, "Vigencia de la resolucion", "no se extrae la resolucion"),
        (9, "Coherencia QR <-> CUFE impreso", "el CUFE impreso no se OCR-ea"),
        (12, "Retenciones <= base", "no se extraen retenciones"),
        (13, "Forma/medio de pago", "no se extrae la forma de pago")]:
        add(n, nombre, "NO_APLICA", motivo)
    return R


# ── capa B: cruce con la BD de referencia ────────────────────────────────────────
def _cargar_db_duckdb(path):
    """Lee la BD con DuckDB (read_csv robusto: maneja comillas, campos multilinea
    del QR, BOM y filas irregulares). Indexa por CUFE (CODIGO_CUFE) normalizado."""
    import duckdb
    p = path.replace("'", "''")          # escapar comillas para inyectar la ruta en el SQL
    con = duckdb.connect()
    try:
        cur = con.execute(
            f"SELECT * FROM read_csv('{p}', delim=';', header=true, all_varchar=true, "
            "quote='\"', escape='\"', ignore_errors=true, null_padding=true)")
        cols = [c[0].lstrip("﻿") for c in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    idx = {}
    for r in rows:
        d = {c: (v if v is not None else "") for c, v in zip(cols, r)}
        k = _norm_cufe(d.get(COL_CUFE_BD, ""))
        if k:
            idx[k] = d            # el ultimo gana si hubiera CUFE repetido (en correccion)
    return idx, 0                 # DuckDB ya descarta/limpia internamente

def _cargar_db_csv(path):
    """Fallback sin DuckDB: csv.reader, descarta filas con columnas != header."""
    idx, descartadas = {}, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=";")
        header = next(r)
        for row in r:
            if len(row) != len(header):
                descartadas += 1
                continue
            d = dict(zip(header, row))
            k = _norm_cufe(d.get(COL_CUFE_BD, ""))
            if k:
                idx[k] = d
    return idx, descartadas

def cargar_db(path, motor="auto"):
    """Indexa la BD de referencia por CUFE (CODIGO_CUFE, normalizado a 96 hex).
    motor: 'duckdb' (robusto, recomendado), 'csv' (fallback), 'auto' (duckdb si esta)."""
    if motor in ("auto", "duckdb"):
        try:
            return _cargar_db_duckdb(path)
        except ImportError:
            if motor == "duckdb":
                raise SystemExit("DuckDB no esta instalado: pip install duckdb (o usa --motor csv)")
            print("(duckdb no disponible; usando lector csv)")
    return _cargar_db_csv(path)

def cruzar_con_db(campos, db_idx, cufe=None, qr_raw=None):
    """Cruza contra la BD POR CUFE. Paso 1 (existencia): busca el CUFE extraido del QR en la
    base; si no esta -> encontrada=False (la factura no existe / es falsa, no se comparan
    campos). Paso 2 (coherencia): si esta, confronta los demas campos contra el registro."""
    k = _norm_cufe(_cufe_de(cufe, qr_raw) or "")
    row = db_idx.get(k) if k else None
    if row is None:
        return {"encontrada": False, "llave": k, "campos": []}   # CUFE no existe en la base
    detalle, n_dif, n_ok = [], 0, 0
    for campo, cmp in COMPARADORES.items():
        # Ya emparejado por CUFE (existencia). Aqui se comparan los 8 campos KIE; NUMERO_FACTURA
        # ya no es la llave, asi que se compara como un campo mas. Los que la BD tenga vacios
        # (p.ej. los del emisor, que esta BD no trae) salen NO_COMPARABLE.
        extraido = (campos.get(campo) or "").strip()
        esperado = (row.get(campo) or "").strip()
        if not esperado or esperado.lower() == "null":
            estado = "NO_COMPARABLE"                  # la BD no tiene ese dato (p.ej. emisor)
        else:
            res = cmp(extraido, esperado)
            if res is None:
                estado = "NO_COMPARABLE"
            elif res:
                estado = "COINCIDE"; n_ok += 1
            else:
                estado = "DIFIERE"; n_dif += 1
        detalle.append({"campo": campo, "extraido": extraido, "esperado": esperado, "estado": estado})
    return {"encontrada": True, "llave": k, "campos": detalle, "n_difiere": n_dif, "n_coincide": n_ok}


# ── validez por campo (sin veredicto global) ─────────────────────────────────────
def validez_por_campo(cruce):
    """A partir del cruce con la BD, devuelve la validez por factura:
       validez = COINCIDE / (COINCIDE + DIFIERE)  en [0,1], o None si no hay comparables.
       Tambien el detalle 1/0/None por campo. No emite veredicto."""
    if cruce is None or not cruce.get("encontrada"):
        return {"validez": None, "n_comparables": 0, "n_coincide": 0, "n_difiere": 0, "por_campo": {}}
    por_campo, ok, dif = {}, 0, 0
    for c in cruce["campos"]:
        if c["estado"] == "COINCIDE":
            por_campo[c["campo"]] = 1.0; ok += 1
        elif c["estado"] == "DIFIERE":
            por_campo[c["campo"]] = 0.0; dif += 1
        else:
            por_campo[c["campo"]] = None                  # NO_COMPARABLE
    comparables = ok + dif
    validez = round(ok / comparables, 3) if comparables else None
    return {"validez": validez, "n_comparables": comparables,
            "n_coincide": ok, "n_difiere": dif, "por_campo": por_campo}


def validar(pred, db_idx=None):
    campos = pred.get("campos", {})
    reglas = validar_reglas(campos, cufe=pred.get("cufe"), qr_raw=pred.get("qr_raw"))
    cruce = (cruzar_con_db(campos, db_idx, cufe=pred.get("cufe"), qr_raw=pred.get("qr_raw"))
             if db_idx is not None else None)
    return {"id": pred.get("id"), "validez_campos": validez_por_campo(cruce),
            "reglas": reglas, "cruce": cruce}


# ── CLI ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_json")
    g.add_argument("--pred_dir")
    ap.add_argument("--db", default="", help="CSV de referencia (db_facturas.csv). Opcional.")
    ap.add_argument("--motor", choices=["duckdb", "csv", "auto"], default="duckdb",
                    help="Motor de lectura de la BD: duckdb (default; falla si no esta), csv, auto")
    ap.add_argument("--salida", default="", help="Carpeta para <id>.validacion.json (opcional)")
    args = ap.parse_args()

    db_idx = None
    if args.db:
        db_idx, descartadas = cargar_db(args.db, motor=args.motor)
        extra = f" ({descartadas} filas descartadas)" if descartadas else " (lectura DuckDB)"
        print(f"BD de referencia: {len(db_idx)} facturas indexadas por CUFE{extra}")

    files = [args.pred_json] if args.pred_json else sorted(glob.glob(os.path.join(args.pred_dir, "*.pred.json")))
    print(f"facturas a validar: {len(files)}")
    if args.salida:
        os.makedirs(args.salida, exist_ok=True)

    validez_list, n_no_encontradas = [], 0
    for fp in files:
        pred = json.loads(open(fp, encoding="utf-8").read())
        res = validar(pred, db_idx)
        vc = res["validez_campos"]
        v = vc["validez"]
        etq = f"validez {v:.2f} ({vc['n_coincide']}/{vc['n_comparables']})" if v is not None else "sin comparables"
        print("\n" + "=" * 74)
        print(f"{res['id']}  ->  {etq}")
        for r in res["reglas"]:
            if r["estado"] not in ("NO_APLICA",):
                marca = {"PASA": "OK ", "FALLA": "XX ", "OBSERVACION": "!! "}[r["estado"]]
                print(f"   {marca}[regla {r['n']:>2}] {r['regla']}: {r['detalle']}")
        if res["cruce"] is not None:
            c = res["cruce"]
            if not c["encontrada"]:
                cufe_corto = (c['llave'][:16] + "…") if c['llave'] else "(sin CUFE)"
                print(f"   ?? [BD] CUFE {cufe_corto} NO existe en la base -> factura inexistente / no verificable")
                n_no_encontradas += 1
            else:
                print(f"   -- [BD] CUFE existe (la factura es real); cruce de campos: "
                      f"{c['n_coincide']} coinciden, {c['n_difiere']} difieren")
                for d in c["campos"]:
                    if d["estado"] == "DIFIERE":
                        print(f"      XX {d['campo']}: extraido='{d['extraido']}'  BD='{d['esperado']}'")
                    elif d["estado"] == "COINCIDE":
                        print(f"      OK {d['campo']}")
                if v is not None:
                    validez_list.append(v)
        if args.salida:
            with open(os.path.join(args.salida, f"{res['id']}.validacion.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 74)
    if validez_list:
        media = sum(validez_list) / len(validez_list)
        perfectas = sum(1 for v in validez_list if v >= 0.999)
        print(f"RESUMEN: {len(validez_list)} facturas con cruce | validez media {media:.3f} | "
              f"100% validas {perfectas} ({perfectas/len(validez_list)*100:.1f}%) | "
              f"con discrepancias {len(validez_list)-perfectas} | no encontradas en BD {n_no_encontradas}")
    else:
        print(f"RESUMEN: sin facturas comparables con la BD (no encontradas: {n_no_encontradas})")


if __name__ == "__main__":
    main()
