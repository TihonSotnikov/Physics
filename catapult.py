"""Калькулятор курсового проекта «Катапульта» (механика, НГУ).

Модель: рычаг на оси, снаряд в чашке, энергия от резинок на хвосте
рычага. Дальность считается аналитически (без численного интегрирования).

Допущения: сопротивлением воздуха пренебрегаем (для снаряда 25 г оно
даёт не более 2 % дальности); трение в оси и гистерезис резинки не
учитываются; резинка линейна и невесома.

Все расчёты внутри ведутся в СИ. Геометрические константы ниже заданы
в мм (так они сняты с CAD) и переводятся в метры в одном месте --
в :func:`lever_point`. Флаги CLI принимают мм и г, перевод в СИ делается
один раз в :func:`parse_args`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np

# --- Геометрия из CAD (мм) --------------------------------------------
# Плоскость симметрии катапульты. Начало O: угол основания со стороны
# чашки, x вдоль основания к стойке оси, z вверх от низа основания.
G = 9.8  # как в методичке
AXLE = (120.0, 106.0)  # ось рычага
BALL_LOCAL = (-130.0, 16.0)  # центр снаряда в системе рычага
ARM_CM_LOCAL = (-43.2, -1.05)  # центр масс рычага в системе рычага
TAIL_Z_LOCAL = -1.5  # ось на хвосте: (tail, -1.5) в системе рычага
TAIL_HOLES = (35, 45, 55)
RODS = {"A": (150.0, 20.0), "B": (165.0, 20.0)}  # нижняя ось резинок
BETAS = (30, 35, 40, 45, 50, 55, 60)  # положения упора, град
COCK_PHI = 20.0  # угол рычага во взведённом положении, град
BALL_DIAMETER = 40.0
ARM_MASS_SOLID = 50.6  # г, сплошной PLA -- заменить взвешенной
ARM_I_SOLID = 2.767e-4  # кг·м², сплошной PLA -- заменить измеренным

# --- Прочие константы (СИ) ---------------------------------------------
Y_LAND = BALL_DIAMETER / 2 / 1000  # м, центр снаряда при касании пола
WINDOW = (1.2, 1.5)  # м, окно дальности из задания


@dataclass(frozen=True)
class Inputs:
    """Входные данные расчёта (СИ).

    Attributes
    ----------
    k : float
        Суммарная жёсткость резинок, Н/м.
    l_free : float
        Свободная длина петли между осями, м.
    tail : int
        Отверстие на хвосте рычага (координата в системе рычага, мм).
    rod : str
        Ключ нижней оси резинок в ``RODS``.
    beta : float
        Положение упора, град.
    m_ball : float
        Масса снаряда, кг.
    m_arm : float
        Масса рычага, кг.
    inertia : float
        Момент инерции рычага относительно оси, кг·м².
    g : float
        Ускорение свободного падения, м/с².
    """

    k: float
    l_free: float
    tail: int
    rod: str
    beta: float
    m_ball: float
    m_arm: float
    inertia: float
    g: float


@dataclass(frozen=True)
class Geometry:
    """Геометрия комбинации (tail, rod, beta), СИ.

    Attributes
    ----------
    l_cocked, l_fired : float
        Расстояние от оси на хвосте до нижней оси резинок во взведённом
        положении и в момент вылета, м.
    h0 : float
        Высота центра снаряда в момент вылета, м.
    x_release : float
        Координата x центра снаряда при вылете в системе O, м.
    dh_ball, dh_arm : float
        Подъём центра снаряда и центра масс рычага от взведённого
        положения до вылета, м.
    r_ball : float
        Радиус от оси до центра снаряда, м.
    theta : float
        Угол вылета к горизонту, рад.
    """

    l_cocked: float
    l_fired: float
    h0: float
    x_release: float
    dh_ball: float
    dh_arm: float
    r_ball: float
    theta: float


@dataclass(frozen=True)
class FlightResult:
    """Результат полёта от точки отрыва до касания.

    Attributes
    ----------
    distance : float
        Дальность по горизонтали от точки отрыва, м.
    time : float
        Время полёта, с.
    apex : float
        Высота вершины траектории (центр снаряда), м.
    v_land : float
        Скорость в момент касания, м/с.
    angle_land : float
        Угол траектории к горизонту при касании (вниз), рад.
    """

    distance: float
    time: float
    apex: float
    v_land: float
    angle_land: float


def lever_point(
    local: tuple[float, float], phi_deg: float
) -> tuple[float, float]:
    """Перевести точку из системы рычага в мировую систему O.

    Parameters
    ----------
    local : tuple of float
        Координаты (xl, zl) точки в системе рычага, мм.
    phi_deg : float
        Угол поворота рычага, град.

    Returns
    -------
    tuple of float
        Координаты (x, z) в системе O, м.
    """
    phi = math.radians(phi_deg)
    xl, zl = local
    x = AXLE[0] + xl * math.cos(phi) - zl * math.sin(phi)
    z = AXLE[1] + xl * math.sin(phi) + zl * math.cos(phi)
    return x / 1000, z / 1000


def combo_geometry(tail: int, rod: str, beta: float) -> Geometry:
    """Посчитать геометрию комбинации (tail, rod, beta).

    Parameters
    ----------
    tail : int
        Отверстие на хвосте рычага, мм.
    rod : str
        Ключ нижней оси резинок в ``RODS``.
    beta : float
        Положение упора, град.

    Returns
    -------
    Geometry
        Геометрия в СИ.
    """
    phi_fired = beta - 90.0
    rod_x, rod_z = RODS[rod][0] / 1000, RODS[rod][1] / 1000
    tail_local = (float(tail), TAIL_Z_LOCAL)

    tail_c = lever_point(tail_local, COCK_PHI)
    tail_f = lever_point(tail_local, phi_fired)
    l_cocked = math.hypot(tail_c[0] - rod_x, tail_c[1] - rod_z)
    l_fired = math.hypot(tail_f[0] - rod_x, tail_f[1] - rod_z)

    ball_c = lever_point(BALL_LOCAL, COCK_PHI)
    ball_f = lever_point(BALL_LOCAL, phi_fired)
    arm_c = lever_point(ARM_CM_LOCAL, COCK_PHI)
    arm_f = lever_point(ARM_CM_LOCAL, phi_fired)

    r_ball = math.hypot(*BALL_LOCAL) / 1000
    # Скорость центра снаряда перпендикулярна радиусу ось -> центр
    # снаряда, а не линии рычага. Центр смещён от линии рычага на
    # atan2(16, 130) ≈ 7.0°, поэтому реальный угол вылета на столько
    # меньше угла упора: θ ≈ β − 7.0°.
    theta = math.radians(beta) - math.atan2(BALL_LOCAL[1], -BALL_LOCAL[0])
    return Geometry(
        l_cocked=l_cocked,
        l_fired=l_fired,
        h0=ball_f[1],
        x_release=ball_f[0],
        dh_ball=ball_f[1] - ball_c[1],
        dh_arm=arm_f[1] - arm_c[1],
        r_ball=r_ball,
        theta=theta,
    )


def rubber_energy(inputs: Inputs, geom: Geometry) -> float:
    """Энергия, отданная резинкой между взведением и вылетом.

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    geom : Geometry
        Геометрия комбинации.

    Returns
    -------
    float
        E = k (x1² − x2²) / 2, Дж. Если резинка провисает до удара
        (``l_fired < l_free``), x2 = 0.

    Raises
    ------
    ValueError
        Если во взведённом положении резинка не натянута (x1 <= 0).
    """
    x1 = geom.l_cocked - inputs.l_free
    x2 = max(geom.l_fired - inputs.l_free, 0.0)
    if x1 <= 0:
        raise ValueError(
            "Резинка не натянута во взведённом положении (x1 <= 0)."
        )
    return 0.5 * inputs.k * (x1**2 - x2**2)


def required_energy(inputs: Inputs, geom: Geometry, v: float) -> float:
    """Энергия резинки, нужная для скорости вылета ``v``.

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    geom : Geometry
        Геометрия комбинации.
    v : float
        Скорость центра снаряда при вылете, м/с.

    Returns
    -------
    float
        Энергия, Дж.
    """
    m_eff = inputs.m_ball + inputs.inertia / geom.r_ball**2
    return (
        0.5 * m_eff * v**2
        + inputs.m_ball * inputs.g * geom.dh_ball
        + inputs.m_arm * inputs.g * geom.dh_arm
    )


def launch_speed(inputs: Inputs, geom: Geometry) -> float:
    """Скорость вылета центра снаряда из баланса энергии.

    E = (m_ball + I / r²) v² / 2 + m_ball g dh_ball + m_arm g dh_arm.

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    geom : Geometry
        Геометрия комбинации.

    Returns
    -------
    float
        Скорость, м/с.

    Raises
    ------
    ValueError
        Если энергии резинки не хватает на подъём снаряда и рычага.
    """
    energy = rubber_energy(inputs, geom)
    lift = (
        inputs.m_ball * inputs.g * geom.dh_ball
        + inputs.m_arm * inputs.g * geom.dh_arm
    )
    kinetic = energy - lift
    if kinetic <= 0:
        raise ValueError("Энергии резинки не хватает на подъём рычага.")
    m_eff = inputs.m_ball + inputs.inertia / geom.r_ball**2
    return math.sqrt(2 * kinetic / m_eff)


def energy_balance(inputs: Inputs, geom: Geometry, v: float) -> dict:
    """Баланс энергии (Р.4).

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    geom : Geometry
        Геометрия комбинации.
    v : float
        Скорость вылета центра снаряда, м/с.

    Returns
    -------
    dict
        Энергии в Дж: ``E_резинки``, ``Eк_снаряда``, ``Eк_рычага``
        (потеря при ударе об упор, теорема Карно: рычаг останавливается
        упором в момент отрыва снаряда), ``Eп_снаряда``, ``Eп_рычага``
        и ``невязка`` = E_резинки − сумма остальных.

    Raises
    ------
    ValueError
        Если резинка не натянута (x1 <= 0).
    """
    omega = v / geom.r_ball
    energy = rubber_energy(inputs, geom)
    ek_ball = 0.5 * inputs.m_ball * v**2
    ek_arm = 0.5 * inputs.inertia * omega**2
    ep_ball = inputs.m_ball * inputs.g * geom.dh_ball
    ep_arm = inputs.m_arm * inputs.g * geom.dh_arm
    return {
        "E_резинки": energy,
        "Eк_снаряда": ek_ball,
        "Eк_рычага": ek_arm,
        "Eп_снаряда": ep_ball,
        "Eп_рычага": ep_arm,
        "невязка": energy - ek_ball - ek_arm - ep_ball - ep_arm,
    }


def flight(
    v: float, theta: float, h0: float, y_land: float, g: float
) -> FlightResult:
    """Аналитический полёт от точки отрыва до касания.

    Старт из (0, h0) со скоростью ``v`` под углом ``theta``; конец --
    центр снаряда на высоте ``y_land``. Сопротивление воздуха не
    учитывается.

    Parameters
    ----------
    v : float
        Начальная скорость, м/с.
    theta : float
        Угол вылета, рад.
    h0 : float
        Начальная высота, м.
    y_land : float
        Высота центра снаряда при касании, м.
    g : float
        Ускорение свободного падения, м/с².

    Returns
    -------
    FlightResult
        Дальность, время, вершина, скорость и угол при касании.

    Raises
    ------
    ValueError
        Если траектория не достигает высоты ``y_land``.
    """
    vx = v * math.cos(theta)
    vy = v * math.sin(theta)
    disc = vy**2 + 2 * g * (h0 - y_land)
    if disc < 0:
        raise ValueError("Траектория не достигает уровня касания.")
    t = (vy + math.sqrt(disc)) / g
    vy_end = vy - g * t
    return FlightResult(
        distance=vx * t,
        time=t,
        apex=h0 + max(vy, 0.0) ** 2 / (2 * g),
        v_land=math.hypot(vx, vy_end),
        angle_land=math.atan2(-vy_end, vx),
    )


def trajectory(
    v: float,
    theta: float,
    h0: float,
    y_land: float,
    g: float,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Массивы x(t), y(t) для графика.

    Parameters
    ----------
    v : float
        Начальная скорость, м/с.
    theta : float
        Угол вылета, рад.
    h0 : float
        Начальная высота, м.
    y_land : float
        Высота центра снаряда при касании, м.
    g : float
        Ускорение свободного падения, м/с².
    n : int, default 200
        Число точек.

    Returns
    -------
    tuple of numpy.ndarray
        Массивы x и y в метрах от точки отрыва.

    Raises
    ------
    ValueError
        Если траектория не достигает высоты ``y_land``.
    """
    t_end = flight(v, theta, h0, y_land, g).time
    t = np.linspace(0.0, t_end, n)
    x = v * math.cos(theta) * t
    y = h0 + v * math.sin(theta) * t - 0.5 * g * t**2
    return x, y


def simulate(inputs: Inputs) -> FlightResult:
    """Прямой расчёт: от входных данных до дальности.

    Parameters
    ----------
    inputs : Inputs
        Входные данные.

    Returns
    -------
    FlightResult
        Результат полёта до касания на высоте ``Y_LAND``.

    Raises
    ------
    ValueError
        Если резинка не натянута или энергии не хватает.
    """
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    v = launch_speed(inputs, geom)
    return flight(v, geom.theta, geom.h0, Y_LAND, inputs.g)


def required_speed(
    x: float, theta: float, h0: float, y_land: float, g: float = G
) -> float:
    """Скорость вылета для заданной дальности (замкнутая форма).

    v² = g x² / (2 cos²θ (h0 − y_land + x tgθ)).

    Parameters
    ----------
    x : float
        Нужная дальность, м.
    theta : float
        Угол вылета, рад.
    h0 : float
        Высота вылета, м.
    y_land : float
        Высота центра снаряда при касании, м.
    g : float, default G
        Ускорение свободного падения, м/с².

    Returns
    -------
    float
        Скорость, м/с.

    Raises
    ------
    ValueError
        Если при таком угле дальность ``x`` недостижима.
    """
    den = 2 * math.cos(theta) ** 2 * (h0 - y_land + x * math.tan(theta))
    if den <= 0:
        raise ValueError("Заданная дальность недостижима при этом угле.")
    return math.sqrt(g * x**2 / den)


def required_k(inputs: Inputs, target: float, y_land: float = Y_LAND) -> float:
    """Жёсткость, дающая дальность ``target`` при текущем ``l_free``.

    Parameters
    ----------
    inputs : Inputs
        Входные данные (поле ``k`` игнорируется).
    target : float
        Нужная дальность, м.
    y_land : float, default Y_LAND
        Высота центра снаряда при касании, м.

    Returns
    -------
    float
        Суммарная жёсткость, Н/м.

    Raises
    ------
    ValueError
        Если дальность недостижима или резинка не натянута.
    """
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    v = required_speed(target, geom.theta, geom.h0, y_land, inputs.g)
    x1 = geom.l_cocked - inputs.l_free
    x2 = max(geom.l_fired - inputs.l_free, 0.0)
    if x1 <= 0:
        raise ValueError(
            "Резинка не натянута во взведённом положении (x1 <= 0)."
        )
    return 2 * required_energy(inputs, geom, v) / (x1**2 - x2**2)


def required_l_free(
    inputs: Inputs, target: float, y_land: float = Y_LAND
) -> float:
    """Свободная длина петли, дающая дальность ``target`` при текущем k.

    При l_free <= l_fired величина x1² − x2² = (l_c − l_f)(l_c + l_f −
    2 l_free) линейна по l_free, поэтому решение явное.

    Parameters
    ----------
    inputs : Inputs
        Входные данные (поле ``l_free`` игнорируется).
    target : float
        Нужная дальность, м.
    y_land : float, default Y_LAND
        Высота центра снаряда при касании, м.

    Returns
    -------
    float
        Свободная длина петли, м.

    Raises
    ------
    ValueError
        Если дальность недостижима или результат даёт провисание
        резинки (l_free > l_fired).
    """
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    v = required_speed(target, geom.theta, geom.h0, y_land, inputs.g)
    energy = required_energy(inputs, geom, v)
    lc, lf = geom.l_cocked, geom.l_fired
    l_free = 0.5 * (lc + lf - 2 * energy / (inputs.k * (lc - lf)))
    if l_free > lf:
        raise ValueError(
            "Нужная свободная длина больше l_fired: резинка провиснет."
        )
    return l_free


def inertia_from_period(
    m_arm: float, d: float, period: float, g: float = G
) -> float:
    """Момент инерции рычага по периоду малых колебаний на оси.

    Физический маятник: I = m g d T² / (4π²).

    Parameters
    ----------
    m_arm : float
        Масса рычага, кг.
    d : float
        Расстояние от оси до центра масс, м. Для рычага из CAD
        ``abs(ARM_CM_LOCAL[0]) / 1000``.
    period : float
        Период малых колебаний, с.
    g : float, default G
        Ускорение свободного падения, м/с².

    Returns
    -------
    float
        Момент инерции относительно оси, кг·м².
    """
    return m_arm * g * d * period**2 / (4 * math.pi**2)


def sensitivity(inputs: Inputs) -> dict[str, float]:
    """Чувствительность дальности (центральные разности).

    Parameters
    ----------
    inputs : Inputs
        Входные данные.

    Returns
    -------
    dict
        Подпись возмущения -> изменение дальности, мм.

    Raises
    ------
    ValueError
        Если прямой расчёт невозможен.
    """

    def dist(**changes: float) -> float:
        return simulate(replace(inputs, **changes)).distance

    def diff(plus: float, minus: float) -> float:
        return (plus - minus) / 2 * 1000

    k, inertia = inputs.k, inputs.inertia
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    v = launch_speed(inputs, geom)

    def fly(v_: float, theta: float) -> float:
        return flight(v_, theta, geom.h0, Y_LAND, inputs.g).distance

    return {
        "+1 мм l_free": diff(
            dist(l_free=inputs.l_free + 1e-3),
            dist(l_free=inputs.l_free - 1e-3),
        ),
        "+1 г m_ball": diff(
            dist(m_ball=inputs.m_ball + 1e-3),
            dist(m_ball=inputs.m_ball - 1e-3),
        ),
        "+1 % k": diff(dist(k=k * 1.01), dist(k=k * 0.99)),
        "+10 % I": diff(
            dist(inertia=inertia * 1.1), dist(inertia=inertia * 0.9)
        ),
        "+1° θ (v, h0 фикс.)": diff(
            fly(v, geom.theta + math.radians(1)),
            fly(v, geom.theta - math.radians(1)),
        ),
        "+1 % v (θ, h0 фикс.)": diff(
            fly(v * 1.01, geom.theta), fly(v * 0.99, geom.theta)
        ),
    }


def combo_table(inputs: Inputs) -> list[str]:
    """Таблица дальности для всех пар (tail, rod) и всех β.

    Parameters
    ----------
    inputs : Inputs
        Входные данные; поля ``tail``, ``rod`` и ``beta`` игнорируются.

    Returns
    -------
    list of str
        Строки таблицы: строка -- β, столбец -- пара (tail, rod).
        Ячейка в окне ``WINDOW`` помечена ``*``.
    """
    pairs = [(t, r) for t in TAIL_HOLES for r in RODS]
    width = 13
    head = "β, °".ljust(6) + "".join(f"{t}/{r}".rjust(width) for t, r in pairs)
    lines = [head]
    for beta in BETAS:
        cells = []
        for tail, rod in pairs:
            case = replace(inputs, tail=tail, rod=rod, beta=beta)
            geom = combo_geometry(tail, rod, beta)
            if geom.l_cocked <= inputs.l_free:
                cell = "мало энергии"
            elif geom.l_fired < inputs.l_free:
                cell = "провис"
            else:
                try:
                    dist = simulate(case).distance
                except ValueError:
                    cell = "мало энергии"
                else:
                    mark = "*" if WINDOW[0] <= dist <= WINDOW[1] else ""
                    cell = f"{dist:.3f}{mark}"
            cells.append(cell.rjust(width))
        lines.append(str(beta).ljust(6) + "".join(cells))
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы CLI и перевести мм и г в СИ.

    Parameters
    ----------
    argv : list of str, optional
        Аргументы командной строки; по умолчанию ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Аргументы; ``l_free`` в м, ``m_ball`` и ``m_arm`` в кг.
    """
    p = argparse.ArgumentParser(description="Калькулятор «Катапульта».")
    p.add_argument("--k", type=float, default=250.0, help="жёсткость, Н/м")
    p.add_argument(
        "--l-free",
        type=float,
        default=50.0,
        help="свободная длина петли, мм (ВРЕМЕННО: измерить)",
    )
    p.add_argument("--tail", type=int, default=35, choices=TAIL_HOLES)
    p.add_argument("--rod", default="A", choices=sorted(RODS))
    p.add_argument("--beta", type=float, default=45.0, help="упор, град")
    p.add_argument("--m-ball", type=float, default=25.0, help="снаряд, г")
    p.add_argument(
        "--m-arm", type=float, default=ARM_MASS_SOLID, help="рычаг, г"
    )
    p.add_argument(
        "--inertia",
        type=float,
        default=ARM_I_SOLID,
        help="момент инерции рычага, кг·м²",
    )
    p.add_argument(
        "--period",
        type=float,
        help="период колебаний рычага, с (переопределяет --inertia)",
    )
    p.add_argument(
        "--target", type=float, default=1.35, help="нужная дальность, м"
    )
    p.add_argument("--k-loop", type=float, help="жёсткость одной петли, Н/м")
    p.add_argument("--save", help="сохранить график в PNG")
    p.add_argument(
        "--no-plot", action="store_true", help="не показывать график"
    )
    args = p.parse_args(argv)
    args.l_free /= 1000
    args.m_ball /= 1000
    args.m_arm /= 1000
    return args


def print_report(
    inputs: Inputs, target: float, k_loop: float | None, period: float | None
) -> None:
    """Напечатать полный отчёт.

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    target : float
        Нужная дальность для обратного режима, м.
    k_loop : float or None
        Жёсткость одной петли, Н/м.
    period : float or None
        Период колебаний рычага, с (если по нему считали I).

    Raises
    ------
    ValueError
        Если прямой расчёт невозможен.
    """
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    x1 = geom.l_cocked - inputs.l_free
    x2 = max(geom.l_fired - inputs.l_free, 0.0)
    v = launch_speed(inputs, geom)
    res = flight(v, geom.theta, geom.h0, Y_LAND, inputs.g)

    print("1. Исходные данные")
    print(f"   k = {inputs.k:.1f} Н/м, l_free = {inputs.l_free * 1e3:.1f} мм")
    print(
        f"   комбинация: tail {inputs.tail}, rod {inputs.rod}, "
        f"β = {inputs.beta:.1f}°"
    )
    print(
        f"   m_ball = {inputs.m_ball * 1e3:.1f} г, "
        f"m_arm = {inputs.m_arm * 1e3:.1f} г, g = {inputs.g} м/с²"
    )
    src = f" (по периоду T = {period} с)" if period else ""
    print(f"   I = {inputs.inertia:.4g} кг·м²{src}")

    print("\n2. Геометрия комбинации")
    print(f"   l_cocked = {geom.l_cocked * 1e3:.1f} мм")
    print(f"   l_fired  = {geom.l_fired * 1e3:.1f} мм")
    print(f"   x1 = {x1 * 1e3:.1f} мм, x2 = {x2 * 1e3:.1f} мм")
    print(f"   h0 = {geom.h0 * 1e3:.1f} мм")
    print(f"   x_release = {geom.x_release * 1e3:.1f} мм")
    print(
        f"   подъём: снаряд {geom.dh_ball * 1e3:.1f} мм, "
        f"рычаг (ЦМ) {geom.dh_arm * 1e3:.1f} мм"
    )
    print(f"   θ = {math.degrees(geom.theta):.2f}° (β − 7.0°)")
    print(f"   r_ball = {geom.r_ball * 1e3:.1f} мм")
    if geom.l_fired < inputs.l_free:
        print(
            "   ВНИМАНИЕ: l_fired < l_free -- резинка провисает до удара "
            "(x2 принят равным 0)."
        )

    print("\n3. Баланс энергии (Р.4)")
    bal = energy_balance(inputs, geom, v)
    for name, val in bal.items():
        note = ""
        if name == "Eк_рычага":
            note = "  (потеря при ударе об упор, теорема Карно)"
        print(f"   {name:<11} = {val:.4f} Дж{note}")
    print(f"   невязка в единицах Дж: {bal['невязка']:.2e}")

    print("\n4. Полёт (Р.5)")
    print(f"   v = {v:.3f} м/с, θ = {math.degrees(geom.theta):.2f}°")
    print(f"   время полёта = {res.time:.3f} с")
    print(f"   вершина = {res.apex * 1e3:.0f} мм над основанием")
    print(f"   дальность = {res.distance:.3f} м")
    print(
        f"   касание: v = {res.v_land:.2f} м/с, "
        f"угол = {math.degrees(res.angle_land):.1f}° вниз"
    )
    hit = WINDOW[0] <= res.distance <= WINDOW[1]
    print(
        f"   окно {WINDOW[0]}-{WINDOW[1]} м: "
        + ("попадает" if hit else "НЕ попадает")
    )

    print(f"\n5. Обратный режим (цель {target:.3f} м)")
    try:
        v_req = required_speed(target, geom.theta, geom.h0, Y_LAND, inputs.g)
        print(f"   нужная v = {v_req:.3f} м/с")
        k_req = required_k(inputs, target)
        print(
            f"   нужная k = {k_req:.1f} Н/м (при l_free = "
            f"{inputs.l_free * 1e3:.1f} мм)"
        )
        if k_loop:
            n = k_req / k_loop
            print(f"   петель: {n:.2f}, округлённо {round(n)}")
        lf_req = required_l_free(inputs, target)
        print(
            f"   нужная l_free = {lf_req * 1e3:.1f} мм (при k = "
            f"{inputs.k:.1f} Н/м)"
        )
    except ValueError as err:
        print(f"   ошибка: {err}")

    print("\n6. Чувствительность дальности")
    for name, val in sensitivity(inputs).items():
        print(f"   {name:<22} {val:+7.1f} мм")

    print("\n7. Таблица комбинаций, дальность, м")
    print(f"   (* -- в окне {WINDOW[0]}-{WINDOW[1]} м)")
    for line in combo_table(inputs):
        print("   " + line)


def plot_trajectory(inputs: Inputs, save: str | None, show: bool) -> None:
    """Построить график траектории (для Р.8).

    Parameters
    ----------
    inputs : Inputs
        Входные данные.
    save : str or None
        Путь для сохранения PNG.
    show : bool
        Показать окно с графиком.

    Raises
    ------
    ValueError
        Если прямой расчёт невозможен.
    """
    geom = combo_geometry(inputs.tail, inputs.rod, inputs.beta)
    v = launch_speed(inputs, geom)
    res = flight(v, geom.theta, geom.h0, Y_LAND, inputs.g)
    x, y = trajectory(v, geom.theta, geom.h0, Y_LAND, inputs.g)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(WINDOW[0], WINDOW[1], alpha=0.2, label="окно 1.2-1.5 м")
    ax.axhline(Y_LAND, color="k", lw=1, label="уровень пола (центр шара)")
    ax.plot(x, y, label="траектория")
    ax.plot(res.distance, Y_LAND, "ro")
    ax.annotate(
        f"{res.distance:.3f} м",
        (res.distance, Y_LAND),
        textcoords="offset points",
        xytext=(5, 10),
    )
    ax.set_xlabel("расстояние от точки отрыва, м")
    ax.set_ylabel("высота центра снаряда, м")
    ax.set_title(
        f"tail {inputs.tail}, rod {inputs.rod}, β = {inputs.beta:.0f}°, "
        f"v = {v:.2f} м/с"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
    if show:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: отчёт и график.

    Parameters
    ----------
    argv : list of str, optional
        Аргументы командной строки.

    Returns
    -------
    int
        Код возврата процесса.
    """
    args = parse_args(argv)
    inertia = args.inertia
    if args.period:
        d = abs(ARM_CM_LOCAL[0]) / 1000
        inertia = inertia_from_period(args.m_arm, d, args.period, G)
    inputs = Inputs(
        k=args.k,
        l_free=args.l_free,
        tail=args.tail,
        rod=args.rod,
        beta=args.beta,
        m_ball=args.m_ball,
        m_arm=args.m_arm,
        inertia=inertia,
        g=G,
    )
    try:
        print_report(inputs, args.target, args.k_loop, args.period)
        if args.save or not args.no_plot:
            plot_trajectory(inputs, args.save, not args.no_plot)
    except ValueError as err:
        print(f"Ошибка расчёта: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
