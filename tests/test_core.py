import numpy as np
import pytest
from PIL import Image

from eggfotos.core import adjustments, blur, filters, geometry
from eggfotos.core.history import History
from eggfotos.core.image_io import load_image, save_image


# ------------------------------------------------------------------ image_io

def test_roundtrip_png(tmp_path, rgb):
    p = save_image(rgb, tmp_path / "a.png")
    back, meta = load_image(p)
    assert np.array_equal(back, rgb)
    assert meta["format"] == "PNG"


def test_rgba_to_jpg_flattens(tmp_path, rgba):
    p = save_image(rgba, tmp_path / "a.jpg")
    back, _ = load_image(p)
    assert back.shape == rgba.shape[:2] + (3,)
    assert back[0, 0].min() > 240  # área transparente virou branco


def test_exif_orientation_applied_and_reset(tmp_path, rgb):
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[0x0112] = 6  # girar 90° horário
    src = tmp_path / "o.jpg"
    im.save(src, exif=exif.tobytes(), quality=95)
    arr, meta = load_image(src)
    assert arr.shape[:2] == (rgb.shape[1], rgb.shape[0])
    out = save_image(arr, tmp_path / "o2.jpg", meta=meta)
    arr2, _ = load_image(out)
    assert arr2.shape == arr.shape  # não girou duas vezes


def test_jpg_high_detail_image(tmp_path):
    """Regressão: ruído forte + optimize=True estourava o buffer do libjpeg."""
    noise = np.random.default_rng(1).integers(0, 255, (800, 1200, 3), dtype=np.uint8)
    for q in (92, 100):
        p = save_image(noise, tmp_path / f"n{q}.jpg", quality=q)
        assert load_image(p)[0].shape == noise.shape


def test_save_unknown_extension(tmp_path, rgb):
    with pytest.raises(ValueError):
        save_image(rgb, tmp_path / "a.xyz")


def test_unicode_path(tmp_path, rgb):
    p = save_image(rgb, tmp_path / "pasta çã" / "foto ação.png")
    assert load_image(p)[0].shape == rgb.shape


# --------------------------------------------------------------- adjustments

def test_adjust_neutral_is_identity(rgb):
    assert np.array_equal(adjustments.adjust(rgb), rgb)


@pytest.mark.parametrize("key", adjustments.ADJUSTMENT_KEYS)
@pytest.mark.parametrize("val", [-100, 50, 100])
def test_adjust_each(rgb, key, val):
    lo, hi = adjustments.adjustment_range(key)
    if not lo <= val <= hi:
        pytest.skip("fora da faixa deste ajuste")
    out = adjustments.adjust(rgb, **{key: val})
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not np.array_equal(out, rgb)


def test_adjust_keeps_alpha(rgba):
    out = adjustments.adjust(rgba, brilho=40)
    assert out.shape == rgba.shape
    assert np.array_equal(out[..., 3], rgba[..., 3])


def test_adjust_unknown_key(rgb):
    with pytest.raises(ValueError):
        adjustments.adjust(rgb, foo=1)


def test_brightness_direction(rgb):
    assert adjustments.adjust(rgb, brilho=50).mean() > rgb.mean()
    assert adjustments.adjust(rgb, exposicao=-50).mean() < rgb.mean()


# ------------------------------------------------------------------- filters

@pytest.mark.parametrize("key", filters.filter_keys())
def test_every_filter_rgb_and_rgba(rgb, rgba, key):
    for img in (rgb, rgba):
        out = filters.apply_filter(img, key, 100)
        assert out.shape == img.shape and out.dtype == np.uint8
    if key != "original":
        assert not np.array_equal(filters.apply_filter(rgb, key), rgb)


def test_filter_intensity_zero_is_identity(rgb):
    assert np.array_equal(filters.apply_filter(rgb, "sepia", 0), rgb)


def test_filter_intensity_blends(rgb):
    full = filters.apply_filter(rgb, "pb", 100).astype(int)
    half = filters.apply_filter(rgb, "pb", 50).astype(int)
    expected = (rgb.astype(int) + full) / 2
    assert np.abs(half - expected).max() <= 1


def test_filter_large_image_fast_enough():
    import time
    big = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
    t = time.perf_counter()
    filters.apply_filter(big, "cartoon")
    assert time.perf_counter() - t < 15


def test_unknown_filter(rgb):
    with pytest.raises(ValueError):
        filters.apply_filter(rgb, "nao_existe")


# ------------------------------------------------------------------ geometry

@pytest.mark.parametrize("ratio", geometry.RATIO_CHOICES)
def test_crop_to_ratio(rgb, ratio):
    out = geometry.crop_to_ratio(rgb, ratio)
    a, b = map(float, ratio.split(":"))
    assert abs(out.shape[1] / out.shape[0] - a / b) < 0.03
    assert out.shape[0] <= rgb.shape[0] and out.shape[1] <= rgb.shape[1]


def test_rotate_and_flip(rgb):
    r = geometry.rotate_angle(rgb, 90)
    assert r.shape == (rgb.shape[1], rgb.shape[0], 3)
    assert np.array_equal(r[0, -1], rgb[0, 0])  # canto sup-esq vai para sup-dir
    assert np.array_equal(geometry.rotate_angle(geometry.rotate_angle(rgb, 180), 180), rgb)
    assert np.array_equal(geometry.flip(geometry.flip(rgb)), rgb)


def test_straighten_smaller_no_borders(rgb):
    out = geometry.straighten(rgb, 10)
    assert out.shape[0] < rgb.shape[0] and out.shape[1] < rgb.shape[1]
    assert np.array_equal(geometry.straighten(rgb, 0), rgb)


def test_resize_long_side(rgb):
    assert max(geometry.resize_long_side(rgb, 100).shape[:2]) == 100
    assert geometry.resize_long_side(rgb, 1000).shape == rgb.shape  # não amplia
    assert max(geometry.resize_long_side(rgb, 400, ampliar=True).shape[:2]) == 400


def test_crop_relative_and_clamp(rgb):
    assert geometry.crop_relative(rgb, 0.5, 0.5, 0.5, 0.5).shape == (60, 100, 3)
    assert geometry.crop(rgb, -10, -10, 50, 50).shape == (40, 40, 3)
    with pytest.raises(ValueError):
        geometry.crop(rgb, 500, 500, 10, 10)


def test_downscale(rgb):
    small, s = geometry.downscale(rgb, 100)
    assert max(small.shape[:2]) == 100 and abs(s - 0.5) < 1e-9
    same, s1 = geometry.downscale(rgb, 1000)
    assert same is rgb and s1 == 1.0


# ---------------------------------------------------------------------- blur

@pytest.mark.parametrize("tipo", list(blur.BLUR_TYPES))
def test_blur_only_inside_region(rgb, tipo):
    out = blur.blur_region(rgb, 0.3, 0.33, 0.3, 0.33, tipo=tipo, intensidade=80)
    assert not np.array_equal(out, rgb)
    assert np.array_equal(out[:20, :20], rgb[:20, :20])  # canto longe da região intacto


def test_blur_empty_mask(rgb):
    out = blur.blur_with_mask(rgb, np.zeros(rgb.shape[:2], np.uint8))
    assert np.array_equal(out, rgb)


def test_blur_rgba_keeps_alpha(rgba):
    out = blur.blur_region(rgba, 0, 0, 1, 1, tipo="solido", cor="#ff0000")
    assert np.array_equal(out[..., 3], rgba[..., 3])
    assert tuple(out[60, 100, :3]) == (255, 0, 0)


def test_stroke_mask(rgb):
    m = blur.stroke_mask(rgb.shape, [(10, 10), (100, 10)], 6)
    assert m[10, 50] == 255 and m[60, 50] == 0


# ------------------------------------------------------------------- history

def test_history_undo_redo_steps(rgb):
    h = History()
    h.reset(rgb)
    a = rgb + 1
    b = rgb + 2
    h.push(a, "A", {"op": "girar", "angulo": 90})
    h.push(b, "B", None)
    assert h.steps() == ([{"op": "girar", "angulo": 90}], 1)
    assert h.undo() is a and h.can_redo()
    assert h.steps() == ([{"op": "girar", "angulo": 90}], 0)
    h.push(rgb, "C", {"op": "espelhar", "direcao": "vertical"})
    assert not h.can_redo()  # o ramo B foi descartado
    assert h.undo() is a and h.undo() is rgb and not h.can_undo()


def test_history_trim_keeps_steps(rgb):
    h = History(max_states=3)
    h.reset(rgb)
    for i in range(6):
        h.push(rgb.copy(), str(i), {"op": "girar", "angulo": 90})
    assert len(h._entries) == 3
    assert h.original is rgb
    steps, manual = h.steps()
    assert len(steps) == 6 and manual == 0  # nenhum passo perdido da receita


def test_denoise_reduces_grain_keeps_edges():
    rng = np.random.default_rng(0)
    clean = np.zeros((120, 160, 3), np.uint8)
    clean[:, 80:] = 200
    noisy = np.clip(clean + rng.normal(0, 18, clean.shape), 0, 255).astype(np.uint8)
    out = adjustments.adjust(noisy, ruido=80)
    assert out[:, :60].astype(float).std() < noisy[:, :60].astype(float).std() * 0.5
    assert out[:, 100:].mean() - out[:, :60].mean() > 150  # a borda continua nítida


def test_denoise_negative_is_ignored(rgb):
    assert np.array_equal(adjustments.adjust(rgb, ruido=-50), rgb)


def test_solid_blur_with_original_color(rgb):
    out = blur.blur_region(rgb, 0.3, 0.3, 0.2, 0.2, tipo="solido", cor="#00ff00", cor_original=True)
    h, w = rgb.shape[:2]
    y, x = int(0.4 * h), int(0.4 * w)
    area = rgb[int(0.3 * h):int(0.5 * h), int(0.3 * w):int(0.5 * w)].reshape(-1, 3).mean(axis=0)
    assert np.abs(out[y, x].astype(float) - area).max() <= 1
