"""The flex plant (scripts/debug/make_flex_model.py): an undamped floating base, servo play the encoders
cannot see, and its stiffness randomised per episode."""

import mujoco
import numpy as np
import yaml

from src.environments import make_standing_env

XML = "models/humanoid_real_v2_soleplate_flex.xml"
SRC = "models/humanoid_real_v2_compliant_soleplate.xml"
CFG = yaml.safe_load(open("config/real_humanoid_stand.yaml"))["standing"]


def _flex(m):
    return [j for j in range(m.njnt) if m.joint(j).name.startswith("ankle_flex")]


def test_the_floating_base_has_no_drag_and_the_robot_weighs_the_same():
    m, src = mujoco.MjModel.from_xml_path(XML), mujoco.MjModel.from_xml_path(SRC)
    assert np.all(m.dof_damping[:6] == 0) and np.all(m.dof_armature[:6] == 0)
    assert np.all(src.dof_damping[:6] == 1.0)          # the inherited default this fixes
    assert abs(m.body_mass.sum() - src.body_mass.sum()) < 1e-6
    assert len(_flex(m)) == 2 and m.nu == src.nu


def test_the_play_is_invisible_to_the_policy():
    env = make_standing_env(config=dict(CFG, debug_print_every=0, xml_file=XML))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (228,) and env.action_space.shape == (17,)
    m = env.env.unwrapped.model
    act_joints = {int(m.actuator_trnid[a, 0]) for a in range(m.nu)}
    assert not act_joints & set(_flex(m))


def test_the_play_stiffness_is_drawn_per_episode_and_only_when_asked():
    env = make_standing_env(config=dict(CFG, debug_print_every=0, xml_file=XML,
                                        flex_rand_range=[0.5, 2.0], flex_damping_rand_range=[1.0, 1.0]))
    m = env.env.unwrapped.model
    k0 = mujoco.MjModel.from_xml_path(XML).jnt_stiffness[_flex(m)]
    seen = []
    for s in range(6):
        env.reset(seed=s)
        k = m.jnt_stiffness[_flex(m)]
        assert k[0] == k[1] and 0.5 * k0[0] - 1e-9 <= k[0] <= 2.0 * k0[0] + 1e-9
        seen.append(k[0])
    assert len(set(np.round(seen, 6))) > 1                       # relative to nominal, not compounding
    plain = make_standing_env(config=dict(CFG, debug_print_every=0, xml_file=XML))
    plain.reset(seed=0)
    np.testing.assert_array_equal(plain.env.unwrapped.model.jnt_stiffness[_flex(m)], k0)
