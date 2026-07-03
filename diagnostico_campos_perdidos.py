#!/usr/bin/env python3
"""
diagnostico_campos_perdidos.py
Replica la asignacion token->mejor_caja de preparar_dataset_kie.py y, para cada caja
'campo perdido', explica POR QUE se perdio:
  (A) NINGUN token solapa la caja (ov_max==0)  -> campo es logo/imagen o el OCR no lo leyo
                                                   (irrecuperable sin re-OCR focal)
  (B) SI hay token(s) que solapan la caja, pero su MEJOR caja es otra (token robado por
      una caja vecina con mayor solapamiento) -> colision de cajas sobre un token fundido
      (recuperable: permitir que un token alimente varias cajas, o desempate por contencion)

Reusa exactamente la geometria de preparar_dataset_kie.py.

Uso:
  python diagnostico_campos_perdidos.py \
      --ocr_dir ~/agente-documentos/facturas_ocr \
      --ls_json ~/agente-documentos/label_studio_import.json \
      [--solo_entidad NIT_EMISOR] [--containment 0.5] [--por_emisor 1]
"""
import os, re, json, glob, argparse
from collections import defaultdict
from preparar_dataset_kie import cargar_ls, _ls_a_pixeles, _containment, _cx, _cy

def overlap_sim(tb, fb):
    return max(_containment(tb, fb), _containment(fb, tb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_dir", required=True)
    ap.add_argument("--ls_json", required=True)
    ap.add_argument("--solo_entidad", default="")
    ap.add_argument("--containment", type=float, default=0.5)
    ap.add_argument("--por_emisor", type=int, default=1)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    ls_por_stem = cargar_ls(args.ls_json)
    ocr_jsons = sorted(f for f in glob.glob(os.path.join(args.ocr_dir, "*.json"))
                       if not os.path.basename(f).startswith("_"))

    contadores = defaultdict(lambda: {"A_sin_token": 0, "B_robado": 0})
    vistos = set()
    for jpath in ocr_jsons:
        stem = os.path.splitext(os.path.basename(jpath))[0]
        if stem not in ls_por_stem:
            continue
        fac = json.loads(open(jpath, encoding="utf-8").read())
        tokens = fac.get("tokens", [])
        W = fac.get("img_width"); H = fac.get("img_height")
        if not tokens or not W or not H:
            continue
        boxes = ls_por_stem[stem]
        cajas_px = [_ls_a_pixeles(b, W, H) for b in boxes]

        # replica EXACTA: cada token -> su mejor caja (overlap simetrico)
        mejor = [(-1, 0.0)] * len(tokens)
        for ti, t in enumerate(tokens):
            for bi, fb in enumerate(cajas_px):
                ov = overlap_sim(t["bbox"], fb)
                if ov > mejor[ti][1]:
                    mejor[ti] = (bi, ov)
        # cajas que recibieron al menos un token sobre umbral
        recibidas = set(bi for ti, (bi, ov) in enumerate(mejor)
                        if bi >= 0 and ov >= args.containment)

        emisor = (re.match(r"(layout_\d+)_", stem) or [None, stem])[1]
        for bi, b in enumerate(boxes):
            ent = b["entity"]
            if args.solo_entidad and ent != args.solo_entidad:
                continue
            if bi in recibidas:
                continue  # esta caja no se perdio
            fb = cajas_px[bi]
            # tokens que SI solapan esta caja perdida
            solapan = []
            for ti, t in enumerate(tokens):
                ov = overlap_sim(t["bbox"], fb)
                if ov > 0:
                    solapan.append((ov, ti))
            solapan.sort(reverse=True)
            causa = "A_sin_token" if not solapan else "B_robado"
            contadores[ent][causa] += 1

            key = (emisor, ent)
            if args.por_emisor and key in vistos:
                continue
            vistos.add(key)
            print("="*80)
            print(f"{stem} | perdido: {ent} | causa: {causa}")
            print(f"  caja LS(px): [{fb[0]:.0f},{fb[1]:.0f},{fb[2]:.0f},{fb[3]:.0f}]")
            if not solapan:
                print("  -> NINGUN token solapa la caja (logo/imagen o no leido por OCR)")
            else:
                print("  -> tokens que SOLAPAN esta caja, pero fueron asignados a otra:")
                for ov, ti in solapan[:args.top]:
                    bi2, ov2 = mejor[ti]
                    ganadora = boxes[bi2]["entity"] if bi2 >= 0 else "(ninguna)"
                    print(f"     ov_con_perdida={ov:.2f}  token={tokens[ti]['text']!r}")
                    print(f"        -> se asigno a '{ganadora}' (ov={ov2:.2f})")

    print("\n" + "#"*80)
    print("RESUMEN por entidad (todas las facturas):")
    for ent, c in sorted(contadores.items()):
        print(f"  {ent:<26} A(sin token)={c['A_sin_token']:<4} B(robado por otra caja)={c['B_robado']}")

if __name__ == "__main__":
    main()
