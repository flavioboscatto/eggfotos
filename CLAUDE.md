# CLAUDE.md — EggFotos

Editor de fotos desktop em Python + PyQt6, uso pessoal do autor.
Interface e mensagens em **português do Brasil**.

## Executar e testar

macOS / Linux (ambiente atual do autor):

```bash
source .venv/bin/activate
python main.py
pytest -q
```

Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`, depois os mesmos comandos.

Rode `pytest -q` depois de qualquer mudança em `eggfotos/core` ou `eggfotos/pipeline`.

## Arquitetura

```
main.py                    entrada (freeze_support é obrigatório: o lote usa processos)
eggfotos/
  core/                    processamento puro, SEM Qt
    image_io.py            load_image / save_image (Pillow, EXIF, ICC, gravação atômica)
    imgutils.py            split/merge alpha, sigma relativo, unsharp, vinheta
    adjustments.py         adjust(img, exposicao=…, brilho=…, …)  valores −100..100
    filters.py             registro FILTERS (28, incl. 10 vintage); apply_filter(img, nome, intensidade)
                           auxiliares vintage: _grain, _curves, _duotone, _dust
    background.py          remover fundo: compute_matte (rembg/ONNX) → apply_matte (refino,
                           descontaminação, fundo transparente/cor); remove_background = as duas
                           blur_background / blur_background_auto: desfoca o que não é o objeto
    faces.py               YuNet (cv2.FaceDetectorYN): detect_faces (caixas relativas + score) →
                           blur_boxes (sensibilidade, margem, tipo); blur_faces = as duas
    geometry.py            crop, crop_relative, crop_to_ratio, rotate, straighten, resize, downscale
    blur.py                blur_with_mask, blur_region (coords relativas), stroke_mask
    history.py             desfazer/refazer; cada estado guarda o "step" que o gerou
  pipeline/                SEM Qt
    operations.py          REGISTRY de operações com esquema de parâmetros (Param)
    recipe.py              Recipe (passos + OutputSettings) ⇄ JSON
    runner.py              BatchRunner (ProcessPool, janela deslizante, cancelamento, log CSV)
  ui/                      PyQt6 — só monta parâmetros e chama core/pipeline
    main_window.py         janela, menus, prévias, commit no histórico
    canvas.py              QGraphicsView: zoom, recorte, pincel, região
    panels.py              Ajustes, Filtros (miniaturas), Recortar, Borrar, Fundo, Rostos, ResizeDialog
    batch_dialog.py        diálogo de lote + StepEditor gerado a partir dos Params
    workers.py             QThreads (miniaturas, prévia do lote, lote, TaskWorker genérico)
tests/                     pytest do core e do pipeline
```

## Regras do projeto (importante manter)

1. **Imagem = `np.ndarray` uint8 RGB (H,W,3) ou RGBA (H,W,4).** Toda função do core
   devolve uma nova imagem e nunca altera a entrada. Preservar o canal alpha
   (use `split_alpha` / `merge_alpha`).
2. **Uma função, dois usos.** A edição manual e o lote chamam as mesmas funções do core.
   Ferramenta nova reproduzível em lote = função no `core` + `register(Operation(...))`
   em `pipeline/operations.py`. O diálogo de lote monta o formulário sozinho.
3. **Raios e tamanhos relativos ao tamanho da foto** (`rel_sigma`, `rel_odd`),
   nunca em pixels fixos. É isso que faz a prévia (cópia reduzida de até 1800 px)
   ficar igual ao resultado em resolução total.
4. **Coordenadas relativas (0–1) nos steps** de recorte/região, para a mesma receita
   servir em fotos de tamanhos diferentes. Na cena do canvas, coordenadas = pixels reais.
5. **Histórico:** `MainWindow.commit(img, rótulo, step)`; `step=None` para edições
   manuais não reproduzíveis (ex.: pincel).
6. **Threads:** só arrays e objetos Python atravessam threads; `QPixmap` apenas na thread
   principal. O `core` e o `pipeline` não podem importar PyQt6 (o lote roda em processos).
7. **Originais nunca são sobrescritos pelo lote** (validado em `runner.validate_job`).
8. Erros por foto no lote vão para o log CSV; o lote continua.
9. **Filtros com aleatoriedade (grão, poeira) usam semente fixa e tamanho relativo**:
   mesma foto → mesmo resultado, e a prévia reduzida fica igual ao resultado final.
10. **Operações pesadas (IA) em duas etapas**: parte cara cacheável (`compute_matte`) separada
    da parte rápida e ajustável (`apply_matte`). A UI roda a cara em `TaskWorker` e refina ao vivo.
    O `core` nunca baixa modelo sozinho: `compute_matte` levanta `ModelNotDownloadedError`;
    só `download_model` baixa, e a UI pede confirmação antes.

## Pendências conhecidas da fase atual

- Operações em resolução total (ajustes, filtros, recorte…) ainda rodam na thread da UI com
  cursor de espera (`_busy`). Só a aba Fundo usa QThread (`_run_task`, sem barra de progresso real).
  Migrar o restante para `_run_task`.
- Remoção de fundo: a descontaminação é própria (convolução normalizada), não usa pymatting
  (que o rembg instala, mas não é importado). Se a borda de cabelo/pelo ficar fraca, avaliar
  `pymatting.estimate_foreground_ml` (melhor, porém lento e com JIT na 1ª execução).
- Lote com fundo transparente e saída JPG: o alpha é achatado em branco. Use PNG/WEBP.
- Seleção de recorte: dá para desenhar e mover; ainda não há alças para redimensionar pelas bordas.
- PNG/TIFF de 16 bits são convertidos para 8 bits na abertura.

## Roteiro (fases)

| Fase | Entrega | Status |
|---|---|---|
| 1–3 | Ajustes, filtros, recorte/giro, borrar, histórico, lote e receitas | feito |
| — | Filtros vintage e de foto antiga (10) | feito |
| 4 | Remoção de fundo (rembg: u2netp / isnet-general-use / BiRefNet-lite), refino de borda, descontaminação de cor, fundo transparente ou cor sólida, aba **Fundo**, operação `remover_fundo` no lote | feito, falta testar com foto real |
| 5 | Desfoque de fundo (opção na aba Fundo, reusa `compute_matte`) e desfoque automático de rostos (aba **Rostos**, YuNet ~230 KB, ops `desfocar_fundo` e `desfocar_rostos`) | feito, falta testar com foto real |
| 6 | Marca d'água: máscara por pincel + `cv2.inpaint` (Telea), depois LaMa via ONNX | — |
| 7 | MobileSAM: seleção com um clique para máscara e fundo | — |
| 8 | Marca d'água em lote: máscara ancorada + estimativa multi-imagem / inversão de alpha | — |
| 9 | (Opcional) inpainting por difusão com GPU (torch, diffusers) | — |

Modelos ONNX: baixados sob demanda (botão nas abas Fundo e Rostos) para `~/.eggfotos/modelos`
(`%USERPROFILE%\.eggfotos\modelos` no Windows). Não versionar arquivos de modelo no projeto.
Testes usam as fixtures `fake_matte` e `fake_faces` (conftest) — sem rede nem modelo. O teste do detector real só roda se o YuNet estiver baixado.
