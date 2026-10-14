"""Seleção por tipo de objeto (SegFormer simulado) e integração com o Fundo e o lote."""
import numpy as np
import pytest

from eggfotos.core import background, objects
from eggfotos.pipeline.operations import apply_step, describe_step, get_operation, normalize_step

H, W = 240, 360


@pytest.fixture
def foto():
    y, x = np.mgrid[0:H, 0:W]
    return np.stack([x * 255 // W, y * 255 // H, (x + y) * 255 // (W + H)], 2).astype(np.uint8)


def _frac(m, box):
    y0, y1, x0, x1 = box
    return float((m[y0:y1, x0:x1] > 127).mean())


ESQ, MEIO, DIR = (100, 230, 5, 115), (100, 230, 130, 230), (100, 230, 250, 355)


def test_categorias_e_chaves():
    assert objects.parse_keys("pessoas, Barcos;predios") == ["pessoas", "barcos", "predios"]
    assert objects.parse_keys("") == []
    with pytest.raises(ValueError):
        objects.parse_keys("dragoes")
    # os pedidos do usuário existem
    for k in ("pessoas", "veiculos", "predios", "barcos"):
        assert k in objects.CATEGORIES
    # nenhuma classe em duas categorias
    todas = [c for cat in objects.CATEGORIES.values() for c in cat.ids]
    assert len(todas) == len(set(todas))


def test_scan_e_presentes(fake_objects, foto):
    om = objects.scan(foto)
    assert om.size == (H, W)
    achados = {k: (f, n) for k, f, n in objects.presentes(om)}
    assert set(achados) == {"pessoas", "predios", "veiculos", "ceu"}
    assert all(n == 1 for _, n in achados.values())
    fracs = [f for _, f, _ in objects.presentes(om)]
    assert fracs == sorted(fracs, reverse=True)


def test_manter_so_pessoas(fake_objects, foto):
    m = objects.select(foto, "pessoas", refinar=False)
    assert m.shape == (H, W) and m.dtype == np.uint8
    assert _frac(m, ESQ) > 0.95
    assert _frac(m, MEIO) < 0.02 and _frac(m, DIR) < 0.02


def test_manter_varios_tipos(fake_objects, foto):
    m = objects.select(foto, "pessoas,veiculos", refinar=False)
    assert _frac(m, ESQ) > 0.95 and _frac(m, DIR) > 0.95 and _frac(m, MEIO) < 0.02


def test_apagar_tira_do_resultado(fake_objects, foto):
    base = np.full((H, W), 255, np.uint8)             # a detecção principal pegou tudo
    m = objects.select(foto, "", "veiculos", base=base, refinar=False)
    assert _frac(m, DIR) < 0.05
    assert _frac(m, ESQ) > 0.95 and _frac(m, MEIO) > 0.95


def test_manter_e_apagar_juntos(fake_objects, foto):
    m = objects.select(foto, ["pessoas", "predios"], ["veiculos"], refinar=False)
    assert _frac(m, ESQ) > 0.95 and _frac(m, DIR) < 0.02


def test_erros(fake_objects, foto):
    with pytest.raises(ValueError):
        objects.select(foto, "pessoas", "pessoas", refinar=False)
    with pytest.raises(ValueError):      # sem manter e sem base
        objects.select(foto, "", "veiculos", refinar=False)
    with pytest.raises(ValueError):
        objects.select(foto, "dragoes", refinar=False)


def test_deterministico_e_nao_altera_entrada(fake_objects, foto):
    antes = foto.copy()
    a = objects.select(foto, "pessoas", refinar=False)
    b = objects.select(foto, "pessoas", refinar=False)
    assert np.array_equal(a, b) and np.array_equal(foto, antes)


def test_resolucao_diferente_da_rede(fake_objects):
    img = np.zeros((181, 403, 3), np.uint8)          # tamanho "quebrado"
    m = objects.select(img, "pessoas", refinar=False)
    assert m.shape == (181, 403)


def test_refino_com_sam_respeita_o_miolo(fake_objects, fake_sam, foto, monkeypatch):
    monkeypatch.setattr(fake_sam, "model_available", lambda: True)
    m0 = objects.select(foto, "pessoas", refinar=False)
    m1 = objects.select(foto, "pessoas", refinar=True)
    assert m1.shape == m0.shape
    assert _frac(m1, (130, 200, 20, 90)) > 0.9        # miolo preservado
    assert (m1 > 127).sum() > 0


def test_refino_ignora_massas_como_ceu_e_predio(fake_objects, fake_sam, foto, monkeypatch):
    monkeypatch.setattr(fake_sam, "model_available", lambda: True)
    chamadas = []
    monkeypatch.setattr(fake_sam, "embed", lambda img: chamadas.append(1) or None)
    objects.select(foto, "predios,ceu", refinar=True)
    assert chamadas == []                              # nem calcula o embedding


def test_modelo_ausente_da_erro_claro(foto, monkeypatch):
    monkeypatch.setattr(objects, "model_available", lambda modelo=objects.DEFAULT_MODEL: False)
    with pytest.raises(background.ModelNotDownloadedError):
        objects.scan(foto)


# -------------------------------------------------------- Fundo e lote

def test_matte_for_sem_tipos_usa_a_deteccao_principal(fake_matte, foto):
    m = background.matte_for(foto, "equilibrado")
    assert m.shape == (H, W) and _frac(m, (H // 4 + 5, 3 * H // 4 - 5, W // 4 + 5, 3 * W // 4 - 5)) > 0.99


def test_remover_fundo_mantendo_so_pessoas(fake_objects, foto):
    out = apply_step(foto, {"op": "remover_fundo", "fundo": "transparente", "manter": "pessoas", "suavizar": 0})
    assert out.shape == (H, W, 4)
    assert out[150:200, 20:90, 3].min() == 255
    assert out[150:200, 260:340, 3].max() == 0       # carro virou fundo


def test_desfocar_fundo_mantendo_veiculos(fake_objects, foto):
    out = apply_step(foto, {"op": "desfocar_fundo", "manter": "veiculos", "intensidade": 80, "suavizar": 0})
    assert out.shape == foto.shape
    assert np.array_equal(out[150:200, 270:330], foto[150:200, 270:330])   # carro nítido
    assert not np.array_equal(out[150:200, 20:90], foto[150:200, 20:90])   # pessoa desfocada


def test_apagar_pessoas_com_deteccao_principal(fake_objects, fake_matte, foto):
    out = apply_step(foto, {"op": "remover_fundo", "fundo": "cor", "cor": "#ff0000", "suavizar": 0,
                            "apagar": "pessoas"})
    # a detecção falsa pega o quadrado central; a pessoa está fora dele, então nada muda no centro
    assert out.shape == foto.shape
    assert tuple(out[5, 5]) == (255, 0, 0)


def test_passo_valida_tipos_e_descreve(fake_objects):
    s = normalize_step({"op": "remover_fundo", "manter": "pessoas", "apagar": "veiculos"})
    assert "pessoas" in describe_step(s).lower() and "apagar" in describe_step(s)
    op = get_operation("remover_fundo")
    assert op.check({"manter": "dragoes"}) is not None
    assert op.check({"manter": "pessoas", "apagar": "pessoas"}) is not None
    assert op.check({"manter": "pessoas"}) is None


def test_check_avisa_modelo_nao_baixado(monkeypatch):
    monkeypatch.setattr(objects, "model_available", lambda modelo=objects.DEFAULT_MODEL: False)
    assert "baixado" in get_operation("remover_fundo").check({"manter": "pessoas"})
    assert get_operation("remover_fundo").check({}) is None      # sem tipos: nada a exigir
