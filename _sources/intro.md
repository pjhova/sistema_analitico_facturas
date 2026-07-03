# Sistema Analítico de Facturas Electrónicas

El notebook `sistema_analitico_facturas` orquesta el flujo completo:

- Análisis exploratorio de la base de referencia, del OCR y del balance del dataset.
- OCR con PaddleOCR y lectura del CUFE por código QR.
- Pre-etiquetado con un LLM local (Qwen 2.5) y anotación en Label Studio.
- Construcción del dataset KIE con partición por emisor.
- Entrenamiento y comparación de LayoutLMv3 y LiLT, mejoras (focal loss y ensamble)
  y validación estadística.
- Inferencia y validación por campo (reglas + cruce por CUFE con la base de referencia).
