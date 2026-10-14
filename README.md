# EggFotos

Editor de fotos desktop (uso pessoal) com ajustes, filtros com miniaturas,
recorte, endireitar, ferramenta de borrar e **processamento em lote** por pasta.

## Instalação (Windows / PowerShell)

```powershell
cd "$HOME\Documents\EggFotos"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

> Se o PowerShell bloquear o `Activate.ps1`, rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Executar

```powershell
python main.py
python main.py "C:\caminho\foto.jpg"   # já abre uma foto
```

## Ícone

O ícone fica em `assets/icon` (macOS `.icns`, Windows `.ico` e PNGs) e é gerado a partir do logo:

```bash
python tools/gerar_icones.py --padrao ambar   # ou: claro
```

## Testes

```powershell
pytest -q
```

Os testes cobrem todo o núcleo (`eggfotos/core`) e o lote (`eggfotos/pipeline`).

## O que já funciona (fases 1 a 8 + molduras)

| Área | Recursos |
|---|---|
| Arquivo | Abrir (JPG, PNG, WEBP, BMP, TIFF), arrastar e soltar, Salvar / Salvar como, qualidade JPG/WEBP, mantém EXIF e perfil de cor, corrige a orientação da câmera |
| Ajustes | Exposição, brilho, contraste, realces, sombras, saturação, temperatura, nitidez, **remover granulação**, com prévia ao vivo |
| Filtros | 28 filtros (inclui 10 vintage/foto antiga: Foto antiga, Sépia envelhecido, Daguerreótipo, Polaroid, Anos 70, Kodachrome, Cross process, Filme P&B clássico, Filme 8mm, Papel amarelado) com **miniatura da própria foto**, intensidade 0–100% e prévia |
| Recortar | Livre ou proporção fixa (1:1, 4:3, 3:2, 16:9…), mover a seleção, Enter aplica |
| Girar | 90° esquerda/direita, espelhar, endireitar (−45° a +45° com recorte automático), redimensionar |
| Borrar | Pincel, retângulo ou elipse; gaussiano, mosaico ou tarja sólida (paleta de cores com RGB, ou cor original da área) |
| Fundo | Remover fundo com IA (3 modelos), afinar/suavizar borda, descontaminar cor, fundo transparente, cor sólida ou **desfocado** (efeito retrato); modelo baixado uma vez, sob demanda; entra no lote |
| Seleção por clique | Clique no objeto (MobileSAM) para escolher o que fica no Fundo, ou o que apagar na Marca d'água; Alt+clique / botão direito tira da seleção |
| Objetos por tipo | Na aba Fundo: **reconhece pessoas, carros e veículos, prédios e casas, barcos, animais, árvores, céu, água, montanhas e ruas** (modelo baixado sob demanda). Para cada tipo encontrado escolha **Manter**, **Apagar** ou **Ignorar**; bordas de pessoas e veículos refinadas com o MobileSAM. Entra no lote (passos "Remover fundo" e "Desfocar fundo") |
| Ocultar rosto | Detecta rostos sozinho (YuNet) e aplica mosaico, embaçar ou tarja; sensibilidade e margem ajustáveis; entra no lote |
| Marca d'água | **Perfis prontos** (fundo liso, texto fino, textura, perto de rostos…) e salve os seus. **Ajuda automática sem IA:** "Procurar a marca na foto" acha textos e logotipos e os lista numerados para você **aceitar, aceitar só uma parte ou descartar**; e "Sugerir ajustes" mede o fundo e preenche os controles (com os motivos). Marque com pincel, retângulo ou clique, **aperte a marcação aos traços da marca** (ajuste automático) e remova com preenchimento rápido (Telea/Navier-Stokes) ou por IA (LaMa, modelo baixado sob demanda); um retângulo fixo entra no lote. **Em lote:** salve a marcação como modelo (ancorada ao canto, escala com a largura) ou aprenda uma marca translúcida repetida de uma pasta de fotos |
| Molduras | 15 molduras com miniatura da sua foto: borda, passe-partout, Polaroid, galeria, cartão com sombra, fita, álbum antigo, barras de cinema, **rolo de filme 35 mm, negativo colorido, slide, Super 8, filmadora VHS (REC/PLAY/data) e TV de tubo**; espessura, cor e legenda; entra no lote |
| Histórico | Desfazer / refazer (Ctrl+Z / Ctrl+Y), botão "Original" (clique para ver a foto original, clique de novo para voltar) |
| Lote | Pasta de entrada (com subpastas), receita de passos, prévia em 3 fotos, formato de saída, sufixo, processos paralelos, pular já processadas, log CSV |
| Receitas | "Editar → Salvar edições como receita" transforma o que você fez na foto aberta numa receita para o lote |

## Atalhos

| Atalho | Ação |
|---|---|
| Ctrl+O / Ctrl+S / Ctrl+Shift+S | Abrir / Salvar / Salvar como |
| Ctrl+Z / Ctrl+Y | Desfazer / Refazer |
| Ctrl+B | Processar pasta (lote) |
| Ctrl+R / Ctrl+Shift+R | Girar 90° direita / esquerda |
| Ctrl+0 / Ctrl+1 | Ajustar à janela / 100% |
| Roda do mouse / pinça / Cmd+rolagem | Zoom |
| Rolagem de dois dedos (trackpad) | Mover a foto |
| + / − (na foto) | Aumentar / diminuir zoom |
| Segurar Espaço | Mão para mover a foto (em qualquer ferramenta) |
| Enter / Esc | Aplicar / cancelar recorte |

## Onde ficam as coisas

- Modelos de IA (fundo e rostos): `~/.eggfotos/modelos` (baixados pelos botões das abas Fundo e Rostos)
- Receitas salvas: `%USERPROFILE%\.eggfotos\receitas\*.json`
- Saída do lote: por padrão `<pasta de entrada>\EggFotos_saida` (os originais nunca são alterados)
- Log de cada lote: `eggfotos_log_AAAAMMDD_HHMMSS.csv` dentro da pasta de saída (abre no Excel)
