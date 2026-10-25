"""Тесты калькулятора «Катапульта»."""

import math
from dataclasses import replace

import pytest

import catapult as c

BETAS = (30, 35, 40, 45, 50, 55, 60)


def make_inputs(**changes: float) -> c.Inputs:
    """Собрать опорный набор входных данных (п. 4 задания)."""
    base = c.Inputs(
        k=250.0,
        l_free=0.0525,
        tail=35,
        rod="A",
        beta=45,
        m_ball=0.025,
        m_arm=0.025,
        inertia=1.4e-4,
        g=9.8,
    )
    return replace(base, **changes)


def test_flight_range() -> None:
    """Дальность по эталонному примеру."""
    res = c.flight(3.39, math.radians(45), 0.209, 0.0, 9.81)
    assert res.distance == pytest.approx(1.352, abs=0.002)


def test_geometry_h0_x_release() -> None:
    """h0 и x_release против CAD."""
    h0 = (226.6, 221.7, 215.9, 209.2, 201.8, 193.7, 184.9)
    xr = (68.9, 58.5, 48.7, 39.4, 30.7, 22.7, 15.4)
    for beta, h, x in zip(BETAS, h0, xr, strict=True):
        geom = c.combo_geometry(35, "A", beta)
        assert geom.h0 * 1000 == pytest.approx(h, abs=0.1)
        assert geom.x_release * 1000 == pytest.approx(x, abs=0.1)


@pytest.mark.parametrize(
    ("tail", "rod", "l_cocked", "l_fired"),
    [
        (35, "A", 96.6, (56.6, 57.6, 58.9, 60.5, 62.5, 64.7, 67.2)),
        (55, "B", 103.7, (42.1, 42.7, 44.2, 46.6, 49.6, 53.2, 57.2)),
    ],
)
def test_geometry_rubber(
    tail: int, rod: str, l_cocked: float, l_fired: tuple[float, ...]
) -> None:
    """Длины резинки против CAD."""
    for beta, lf in zip(BETAS, l_fired, strict=True):
        geom = c.combo_geometry(tail, rod, beta)
        assert geom.l_cocked * 1000 == pytest.approx(l_cocked, abs=0.1)
        assert geom.l_fired * 1000 == pytest.approx(lf, abs=0.1)


PLAN = {
    30: (3.20, 3.44, 3.67),
    35: (3.15, 3.38, 3.59),
    40: (3.14, 3.36, 3.57),
    45: (3.17, 3.39, 3.59),
    50: (3.24, 3.46, 3.66),
    55: (3.35, 3.58, 3.79),
    60: (3.53, 3.76, 3.98),
}


@pytest.mark.parametrize("beta", BETAS)
def test_required_speed_plan(beta: int) -> None:
    """Таблица плана: нужная скорость при θ = β."""
    h0 = c.combo_geometry(35, "A", beta).h0
    for x, v in zip((1.2, 1.35, 1.5), PLAN[beta], strict=True):
        got = c.required_speed(x, math.radians(beta), h0, 0.0, 9.81)
        assert got == pytest.approx(v, abs=0.01)


def test_reference_case() -> None:
    """Опорный расчёт."""
    inp = make_inputs()
    geom = c.combo_geometry(35, "A", 45)
    assert c.rubber_energy(inp, geom) == pytest.approx(0.23529, abs=1e-5)
    assert geom.dh_ball * 1000 == pytest.approx(132.7, abs=0.1)
    assert geom.dh_arm * 1000 == pytest.approx(45.6, abs=0.1)
    assert math.degrees(geom.theta) == pytest.approx(37.98, abs=0.01)
    v = c.launch_speed(inp, geom)
    assert v == pytest.approx(3.3996, abs=1e-3)
    assert c.simulate(inp).distance == pytest.approx(1.3496, abs=1e-3)
    bal = c.energy_balance(inp, geom, v)
    assert bal["Eк_рычага"] == pytest.approx(0.04716, abs=1e-5)


def test_energy_balance_residual() -> None:
    """Невязка баланса энергии."""
    inp = make_inputs()
    geom = c.combo_geometry(35, "A", 45)
    bal = c.energy_balance(inp, geom, c.launch_speed(inp, geom))
    assert abs(bal["невязка"]) < 1e-12


def test_round_trip() -> None:
    """Круговые проверки обратного режима."""
    target = 1.35
    inp = make_inputs()
    k = c.required_k(inp, target)
    got = c.simulate(replace(inp, k=k)).distance
    assert got == pytest.approx(target, abs=1e-3)

    l_free = c.required_l_free(inp, target)
    got = c.simulate(replace(inp, l_free=l_free)).distance
    assert got == pytest.approx(target, abs=1e-3)

    geom = c.combo_geometry(35, "A", 45)
    v = c.required_speed(target, geom.theta, geom.h0, c.Y_LAND, inp.g)
    res = c.flight(v, geom.theta, geom.h0, c.Y_LAND, inp.g)
    assert res.distance == pytest.approx(target, abs=1e-9)


def test_errors() -> None:
    """Нет натяжения и нехватка энергии."""
    with pytest.raises(ValueError):
        c.simulate(make_inputs(l_free=0.2))
    with pytest.raises(ValueError):
        c.simulate(make_inputs(k=5.0))
    with pytest.raises(ValueError):
        c.required_l_free(make_inputs(k=5000.0), 1.35)
