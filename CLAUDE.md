# CLAUDE.md — EggFotos

Editor de fotos desktop em Python + PyQt6, uso pessoal do autor.
Interface e mensagens em **português do Brasil**: os botões padrão do Qt vêm das traduções do próprio
Qt (`install_portuguese` em `ui/app.py`) e os erros de bibliotecas passam por `eggfotos/errors.py::friendly_error`
(também no log CSV do lote). Texto novo na UI sempre em português; erro novo de biblioteca: traduzir lá.

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
assets/icon, assets/logo     ícone do app (EggFotos.png/.ico/.icns + variantes claro/âmbar) e o logo original
tools/gerar_icones.py      regenera os ícones a partir do logo (`--padrao ambar|claro` escolhe o ícone do programa)
main.py                    entrada (freeze_support é obrigatório: o lote usa processos)
eggfotos/
  core/                    processamento puro, SEM Qt
    image_io.py            load_image / save_image (Pillow, EXIF, ICC, gravação atômica)
    imgutils.py            split/merge alpha, sigma relativo, unsharp, vinheta
    adjustments.py         adjust(img, exposicao=…, brilho=…, …)  valores −100..100 (exceto `ruido`: 0..100,
                           ADJUSTMENT_RANGES); denoise = NLM na luminância + suavização do croma
    frames.py              molduras (15): registro FRAMES + apply_frame(img, nome, espessura, cor, cor_padrao, texto);
                           papel/galeria, cinema/filme (rolo 35 mm, negativo, slide, Super 8) e eletrônicos (VHS, TV);
                           tamanhos relativos ao menor lado; aumentam a foto (exceto VHS); saída RGB opaca
    filters.py             registro FILTERS (28, incl. 10 vintage); apply_filter(img, nome, intensidade)
                           auxiliares vintage: _grain, _curves, _duotone, _dust
    background.py          remover fundo: compute_matte (rembg/ONNX) → apply_matte (refino,
                           descontaminação, fundo transparente/cor); remove_background = as duas
                           blur_background / blur_background_auto: desfoca o que não é o objeto
    inpaint.py             remover marca d'água: remove_with_mask (telea/ns via cv2.inpaint, lama via ONNX),
                           remove_region (retângulo relativo, vai pro lote); só processa a região da máscara.
                           LaMa roda em janelas de 512 px REAIS (até 12; marcas enormes caem no método reduzido)
                           e restore_grain recoloca o grão do entorno (semente fixa; `grao`/`grao_forca`). Controles do preenchimento:
                           `expandir`, `raio` (Telea/NS), `contexto` (LaMa, 48–168 px), `emenda` (borda da colagem).
                           snap_to_strokes: aperta uma marcação solta nos traços da marca (fundo por mediana,
                           diferença relativa ao brilho; polaridade clara/escura/auto); só edição manual
    watermark.py           modelos de marca em ~/.eggfotos/marcas (env EGGFOTOS_MARCAS substitui):
                           MaskTemplate (marcação ancorada ao canto, tamanho/margens ÷ largura da foto,
                           + inpaint) e AlphaTemplate (estima α de marca translúcida branca/preta em
                           N fotos e inverte I=(J−α·cor)/(1−α)); ops remover_marca_modelo/_estimada
    presets.py             perfis prontos da remoção de marca (PRESETS: liso, texto fino, textura, detalhes, grande,
                           retângulo solto, grão alto, foto limpa) + perfis do usuário em marcas/perfis.json
    segment.py             MobileSAM (ONNX): embed(img) [~0,5 s, cacheável] → predict(emb, pontos) [instantâneo];
                           o encoder NÃO redimensiona: a foto vai com lado maior = 1024 e o decoder
                           recebe orig_im_size = tamanho real; pontos relativos, rótulo 1/0
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
    canvas.py              QGraphicsView: zoom, recorte, pincel, região, clique (TOOL_POINT, Alt/direito = "fora")
    tool_tabs.py           ToolTabs: botões em grade + QStackedWidget; cada painel dentro de QScrollArea (só rola na
                           vertical) e `compact()` (margens/espaços menores, formulários que quebram rótulos longos)
    panels.py              Ajustes, Filtros (miniaturas), Recortar, Borrar, Fundo, Ocultar rosto, Marca d'água, Molduras, ResizeDialog
    color_picker.py        paleta de cores (amostras + RGB + "Mais cores…"), usada em todos os botões de cor
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
5. **Desfazer em dois níveis:** Ctrl+Z desfaz primeiro as MARCAÇÕES em andamento da aba atual, uma por vez
   (`_mark_undo`: traço/retângulo/clique/aperto/limpar na Marca d'água; cliques do modo Fundo em `_sam_points`),
   e só quando não há mais marcação mexe no histórico da foto. Desfazer uma remoção já aplicada devolve as
   marcações (`_mark_applied`). Marcações são guardadas em bits (`np.packbits`), máx. 60.
6. **Histórico:** `MainWindow.commit(img, rótulo, step)`; `step=None` para edições
   manuais não reproduzíveis (ex.: pincel).
7. **Threads:** o resultado de uma tarefa só entra no histórico se a foto não mudou (`uid`); só arrays e objetos Python atravessam threads; `QPixmap` apenas na thread
   principal. O `core` e o `pipeline` não podem importar PyQt6 (o lote roda em processos).
8. **Originais nunca são sobrescritos pelo lote** (validado em `runner.validate_job`).
9. Erros por foto no lote vão para o log CSV; o lote continua.
10. **Filtros com aleatoriedade (grão, poeira) usam semente fixa e tamanho relativo**:
   mesma foto → mesmo resultado, e a prévia reduzida fica igual ao resultado final.
11. **Operações pesadas (IA) em duas etapas**: parte cara cacheável (`compute_matte`) separada
    da parte rápida e ajustável (`apply_matte`). A UI roda a cara em `TaskWorker` e refina ao vivo.
    O `core` nunca baixa modelo sozinho: `compute_matte` levanta `ModelNotDownloadedError`;
    só `download_model` baixa, e a UI pede confirmação antes.

## Pendências conhecidas da fase atual

- Operações pesadas rodam em segundo plano via `MainWindow._run_task` / `_edit_async` (QThread `TaskWorker`):
  cursor de espera na hora, janelinha com Cancelar após 0,3 s (cancelar descarta o resultado; a conta termina
  sozinha), e `_guard()` bloqueia desfazer/abrir/salvar enquanto roda. Só salvar arquivo ainda é síncrono
  (`_busy`), porque `_confirm_discard`/fechar precisam do resultado na hora. Sem barra de progresso real
  (as contas são uma chamada só de NumPy/OpenCV, não dá para medir).
- Remoção de fundo: a descontaminação é própria (convolução normalizada), não usa pymatting
  (que o rembg instala, mas não é importado). Se a borda de cabelo/pelo ficar fraca, avaliar
  `pymatting.estimate_foreground_ml` (melhor, porém lento e com JIT na 1ª execução).
- Remover granulação: o ruído é em pixels reais, então a prévia (cópia reduzida) parece mais limpa que o resultado; conferir em 100%.
- Seleção por cliques (MobileSAM) não é reproduzível em lote por si só: entra no histórico sem step (como o pincel), a menos que a marcação seja salva como modelo (aba Marca d'água), aí vira `remover_marca_modelo`.
- Marca aprendida (alpha): exige fotos da MESMA proporção e marca na mesma posição; o α sai ~20% subestimado em testes sintéticos, ajustar a Força. Marcas grandes/espessas (> 4% do lado maior) confundem o estimador de piso (`traco_max`).
- Molduras: nova moldura = função `(rgb, e, cor, texto) -> rgb` com `@_register(...)` em `core/frames.py`; aparece
  sozinha na aba (miniatura da foto) e no lote. Prévia da aba usa `set_preview(centered=True)` encolhido para caber.
- Perfis da Marca d'água: valores iniciais, não regras; mexer em qualquer controle volta para "Personalizado".
  Novo perfil pronto = uma linha em `presets.PRESETS` (o teste valida faixas e se a receita aceita).
- Barra lateral: largura inicial 400 px, redimensionável e lembrada (`saveState`); `_slider_row` mostra o valor ao lado
  do slider. Painel novo: use `ToolTabs.addTab` (aplica `compact`) e confira que `minimumSizeHint().width()` cabe
  em ~330 px (há teste em tests/test_ui_layout.py).
- Zoom: roda do mouse = zoom; trackpad com dois dedos = mover; pinça ou Cmd+rolagem = zoom; teclas +/−;
  botões na barra de status. Detecção automática de marca em UMA foto não existe de forma confiável:
  o que há é "apertar aos traços" (assistido) e "aprender de uma pasta" (marca repetida).
- Botão Original é um interruptor (clique liga/desliga); ao editar ou trocar de aba ele desliga sozinho.
- Lote com fundo transparente e saída JPG: o alpha é achatado em branco. Use PNG/WEBP.
- Seleção de recorte: dá para desenhar e mover; ainda não há alças para redimensionar pelas bordas.
- PNG/TIFF de 16 bits são convertidos para 8 bits na abertura.

## Roteiro (fases)

| Fase | Entrega | Status |
|---|---|---|
| 1–3 | Ajustes, filtros, recorte/giro, borrar, histórico, lote e receitas | feito |
| — | Filtros vintage e de foto antiga (10) | feito |
| 4 | Remoção de fundo (rembg: u2netp / isnet-general-use / BiRefNet-lite), refino de borda, descontaminação de cor, fundo transparente ou cor sólida, aba **Fundo**, operação `remover_fundo` no lote | feito, falta testar com foto real |
| 5 | Desfoque de fundo (opção na aba Fundo, reusa `compute_matte`) e desfoque automático de rostos (aba **Ocultar rosto**, YuNet ~230 KB, ops `desfocar_fundo` e `desfocar_rostos`) | feito, falta testar com foto real |
| 6 | Marca d'água: aba **Marca d'água** (pincel/retângulo/clique acumulam uma máscara), Telea/Navier-Stokes sem modelo e LaMa (ONNX ~200 MB, baixado sob demanda); op `remover_marca_regiao` (1 retângulo) no lote | feito e testado com o modelo real |
| 7 | MobileSAM: seleção por cliques na aba **Fundo** (alternativa à detecção automática; vira a máscara do fundo) e ferramenta "Clique" na aba **Marca d'água** | feito e testado com o modelo real |
| — | Molduras (15, estilo papel, cinema, filme, filmadora) com aba própria e operação `moldura` no lote | feito |
| 8 | Marca d'água em lote: modelo de máscara ancorada (`remover_marca_modelo`) e marca translúcida aprendida de uma pasta (`remover_marca_estimada`, inversão de alpha); `Operation.check` valida modelos antes do lote | feito; testado com marcas sintéticas, falta testar com marcas reais |
| 9 | (Opcional) inpainting por difusão com GPU (torch, diffusers) | — |

Modelos ONNX: baixados sob demanda (botão nas abas Fundo, Ocultar rosto e Marca d'água) para `~/.eggfotos/modelos`
(`%USERPROFILE%\.eggfotos\modelos` no Windows). O rembg 2.x grava em `modelos/models/<nome>/<nome>.onnx`
(`model_path` aceita esse layout e o antigo). O download dos rostos usa os certificados do `certifi`
(o Python do macOS não traz certificados raiz). Não versionar arquivos de modelo no projeto.
Testes usam as fixtures `fake_matte` e `fake_faces` (conftest) — sem rede nem modelo. O teste do detector real só roda se o YuNet estiver baixado.
