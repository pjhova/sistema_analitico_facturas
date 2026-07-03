# Notebook (Jupyter Book)

## Contenido del repositorio

- `sistema_analitico_facturas.ipynb` — notebook orquestador.
- `intro.md` — pagina de inicio del libro.
- `_config.yml`, `_toc.yml` — configuracion de Jupyter Book.
- `requirements.txt` — dependencias para construir el libro.
- Scripts que el notebook invoca (para consulta; no se ejecutan al construir):
  `extraer_facturas.py`, `pre_etiquetar_facturas.py`, `preparar_dataset_kie.py`,
  `inferir_factura.py`, `reglas_validacion.py`, `validar_kie_estadistico.py`,
  `diagnostico_campos_perdidos.py`.

## Construir el libro localmente

```bash
pip install -r requirements.txt
jupyter-book build .
# El sitio queda en _build/html/index.html
```

## Publicar en GitHub Pages

```bash
pip install ghp-import
ghp-import -n -p -f _build/html
```
