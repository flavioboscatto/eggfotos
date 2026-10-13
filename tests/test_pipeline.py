from pathlib import Path

import numpy as np
import pytest

from eggfotos.core.image_io import load_image, save_image
from eggfotos.pipeline import operations as ops
from eggfotos.pipeline.recipe import OutputSettings, Recipe
from eggfotos.pipeline.runner import (
    BatchJob, BatchRunner, find_images, output_path_for, preview_recipe,
)


def test_every_operation_default_step_runs(rgb, fake_matte, fake_faces, marks_tmp):
    from eggfotos.core import watermark
    m = np.zeros(rgb.shape[:2], np.uint8)
    m[10:30, 120:190] = 255
    watermark.save_mask_template("t", watermark.make_mask_template(m))
    watermark.save_alpha_template("t", watermark.AlphaTemplate(np.full(rgb.shape[:2], 0.2, np.float32), 255, 10))
    for key, op in ops.REGISTRY.items():
        step = op.new_step()
        if key.startswith("remover_marca_") and "nome" in step:
            step["nome"] = "t"
        out = ops.apply_step(rgb, step)
        assert out.dtype == np.uint8, key


def test_normalize_coerces_types():
    s = ops.normalize_step({"op": "girar", "angulo": "180"})
    assert s == {"op": "girar", "angulo": 180}
    s = ops.normalize_step({"op": "ajustes", "brilho": "12.6"})
    assert s["brilho"] == 13 and s["contraste"] == 0


@pytest.mark.parametrize("step", [
    {"angulo": 90},
    {"op": "nao_existe"},
    {"op": "girar", "angulo": 45},
    {"op": "ajustes", "brilho": 500},
    {"op": "filtro", "nome": "pb", "xyz": 1},
])
def test_invalid_steps(step):
    with pytest.raises(ValueError):
        ops.normalize_step(step)


def test_describe_every_op():
    for op in ops.REGISTRY.values():
        assert op.label in ops.describe_step(op.new_step())


def test_recipe_json_roundtrip(tmp_path, rgb):
    r = Recipe("Teste ç", [
        {"op": "filtro", "nome": "sepia", "intensidade": 70},
        {"op": "redimensionar", "lado_maior": 100},
    ], OutputSettings(formato="png", sufixo="_x"))
    p = r.save(tmp_path / "r.json")
    r2 = Recipe.load(p)
    assert r2.to_dict() == r.to_dict()
    assert np.array_equal(r2.apply(rgb), r.apply(rgb))
    assert max(r2.apply(rgb).shape[:2]) == 100


def test_recipe_validate():
    assert Recipe().validate()  # sem passos
    bad = Recipe(passos=[{"op": "girar", "angulo": 1}], saida=OutputSettings(sufixo="a/b"))
    assert len(bad.validate()) == 2


def _make_folder(root: Path, rgb, n=4):
    (root / "sub").mkdir(parents=True)
    for i in range(n):
        save_image(rgb, root / f"f{i}.jpg")
    save_image(rgb, root / "sub" / "s.png")
    (root / "nota.txt").write_text("x")
    return root


def _job(tmp_path, rgb, **kw):
    entrada = _make_folder(tmp_path / "entrada", rgb)
    recipe = Recipe("t", [{"op": "filtro", "nome": "pb"}, {"op": "redimensionar", "lado_maior": 80}],
                    OutputSettings(formato="mesmo", sufixo="_egg"))
    return BatchJob(entrada, tmp_path / "saida", recipe, **kw)


@pytest.mark.parametrize("workers", [1, 2])
def test_runner_processes_folder(tmp_path, rgb, workers):
    job = _job(tmp_path, rgb, recursivo=True, workers=workers)
    events = []
    report = BatchRunner(job, on_progress=lambda d, t, r: events.append((d, t))).run()
    assert report.total == 5 and report.ok == 5 and report.erros == 0
    assert events[-1] == (5, 5)
    out = tmp_path / "saida" / "sub" / "s_egg.png"
    img, _ = load_image(out)
    assert max(img.shape[:2]) == 80
    assert np.allclose(img[..., 0], img[..., 1], atol=2)  # P&B
    assert Path(report.log_path).exists()
    # segunda execução: tudo pulado
    report2 = BatchRunner(job).run()
    assert report2.pulados == 5 and report2.ok == 0


def test_runner_not_recursive(tmp_path, rgb):
    report = BatchRunner(_job(tmp_path, rgb, recursivo=False, workers=1)).run()
    assert report.total == 4


def test_output_inside_input_is_ignored(tmp_path, rgb):
    job = _job(tmp_path, rgb, recursivo=True, workers=1)
    job.saida = job.entrada / "saida"
    assert BatchRunner(job).run().ok == 5
    job.pular_existentes = False
    assert BatchRunner(job).run().total == 5  # não reprocessa a própria saída


def test_refuses_overwriting_originals(tmp_path, rgb):
    job = _job(tmp_path, rgb)
    job.saida = job.entrada
    job.recipe.saida.sufixo = ""
    with pytest.raises(ValueError, match="sobrescritos"):
        BatchRunner(job).run()


def test_errors_are_logged_not_fatal(tmp_path, rgb):
    job = _job(tmp_path, rgb, workers=1)
    (job.entrada / "quebrada.jpg").write_bytes(b"isto nao e imagem")
    report = BatchRunner(job).run()
    assert report.ok == 4 and report.erros == 1
    assert "quebrada" in Path(report.log_path).read_text(encoding="utf-8-sig")


def test_cancel(tmp_path, rgb):
    job = _job(tmp_path, rgb, workers=1)
    report = BatchRunner(job, should_cancel=lambda: True).run()
    assert report.cancelado and report.ok == 0


def test_format_conversion(tmp_path, rgb):
    job = _job(tmp_path, rgb, workers=1)
    job.recipe.saida.formato = "webp"
    BatchRunner(job).run()
    assert len(list((tmp_path / "saida").glob("*.webp"))) == 4


def test_find_images_and_output_path(tmp_path, rgb):
    job = _job(tmp_path, rgb)
    files = find_images(job.entrada, True)
    assert len(files) == 5
    assert output_path_for(job.entrada / "sub" / "s.png", job) == job.saida / "sub" / "s_egg.png"


def test_preview(tmp_path, rgb):
    job = _job(tmp_path, rgb)
    files = find_images(job.entrada)[:2] + [job.entrada / "nota.txt"]
    res = preview_recipe(files, job.recipe, max_side=64)
    assert len(res) == 3
    assert isinstance(res[0][1], np.ndarray)
    assert isinstance(res[2][1], str)
