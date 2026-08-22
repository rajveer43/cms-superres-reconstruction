"""Tests for the SEMD implementation in ``multiscale_sr.spectral``.

These are not optional. A silently-wrong metric is worse than no metric: it
would let us "confirm" the metric-blindness hypothesis with a number that means
nothing. The suite checks the metric axioms the paper proves (identity,
symmetry, triangle inequality), the invariances it advertises (rotation,
permutation), the one place this adaptation could quietly lose energy
(balancing), and gradient finiteness through the sort.

The load-bearing one is :func:`test_geometry_discriminates_where_pixel_metrics_are_blind`,
which encodes the whole reason this module exists.
"""
from __future__ import annotations

import pytest
import torch

from multiscale_sr.engine import energy_response, peak_metrics
from multiscale_sr.spectral import (
    cumulative_spectral,
    extract_particles,
    pairwise_omega,
    semd_images,
    semd_p2,
    spectral_function,
)


def _cloud(n: int, seed: int, scale: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """A random equal-total-energy particle cloud, as (energies, coords)."""
    g = torch.Generator().manual_seed(seed)
    e = torch.rand(1, n, generator=g, dtype=torch.float64) + 0.1
    e = e / e.sum(dim=1, keepdim=True)  # unit total energy for all clouds
    c = torch.rand(1, n, 2, generator=g, dtype=torch.float64) * scale
    return e, c


def _semd(a, b, **kw) -> float:
    return semd_p2(a[0], a[1], b[0], b[1], **kw).item()


# --------------------------------------------------------------------------- #
# Metric axioms
# --------------------------------------------------------------------------- #
def test_identity_of_indiscernibles():
    """SEMD(A, A) == 0 — the paper's headline property for a true metric."""
    a = _cloud(12, seed=0)
    assert _semd(a, a) == pytest.approx(0.0, abs=1e-12)


def test_symmetry():
    a, b = _cloud(10, seed=1), _cloud(14, seed=2)
    assert _semd(a, b) == pytest.approx(_semd(b, a), rel=1e-10)


def test_positivity_for_distinct_events():
    a, b = _cloud(10, seed=3), _cloud(10, seed=4)
    assert _semd(a, b) > 1e-6


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_triangle_inequality(seed):
    """sqrt(SEMD) is the p=2 Wasserstein distance, so it obeys the triangle
    inequality; SEMD itself is that distance squared."""
    a = _cloud(8, seed=seed)
    b = _cloud(8, seed=seed + 100)
    c = _cloud(8, seed=seed + 200)
    d_ab = _semd(a, b) ** 0.5
    d_bc = _semd(b, c) ** 0.5
    d_ac = _semd(a, c) ** 0.5
    assert d_ac <= d_ab + d_bc + 1e-9


# --------------------------------------------------------------------------- #
# Invariances
# --------------------------------------------------------------------------- #
def test_rotation_invariance():
    """A rigid rotation of both clouds leaves every pairwise distance fixed."""
    a, b = _cloud(10, seed=5), _cloud(10, seed=6)
    before = _semd(a, b)

    theta = torch.tensor(0.7, dtype=torch.float64)
    rot = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
        dtype=torch.float64,
    )
    a_rot = (a[0], a[1] @ rot.T)
    b_rot = (b[0], b[1] @ rot.T)
    assert _semd(a_rot, b_rot) == pytest.approx(before, rel=1e-9)


def test_translation_invariance():
    a, b = _cloud(10, seed=7), _cloud(10, seed=8)
    before = _semd(a, b)
    shift = torch.tensor([3.5, -2.0], dtype=torch.float64).view(1, 1, 2)
    assert _semd((a[0], a[1] + shift), (b[0], b[1] + shift)) == pytest.approx(before, rel=1e-9)


def test_particle_permutation_invariance():
    """Relabelling particles must not change the distance — the spectral
    representation is a function of the *set* of pairwise angles."""
    a, b = _cloud(10, seed=9), _cloud(10, seed=10)
    before = _semd(a, b)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    a_perm = (a[0][:, perm], a[1][:, perm, :])
    assert _semd(a_perm, b) == pytest.approx(before, rel=1e-9)


def test_zero_energy_slots_are_inert():
    """Padding slots (E=0) must contribute nothing regardless of position —
    ``extract_particles`` relies on this to keep shapes static."""
    a, b = _cloud(10, seed=11), _cloud(10, seed=12)
    before = _semd(a, b)
    pad_e = torch.zeros(1, 5, dtype=torch.float64)
    pad_c = torch.full((1, 5, 2), 999.0, dtype=torch.float64)
    a_pad = (torch.cat((a[0], pad_e), 1), torch.cat((a[1], pad_c), 1))
    assert _semd(a_pad, b) == pytest.approx(before, rel=1e-9)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_pairwise_omega_symmetric_zero_diagonal():
    _, c = _cloud(7, seed=13)
    om = pairwise_omega(c)
    assert torch.allclose(om, om.transpose(1, 2))
    assert torch.all(torch.diagonal(om, dim1=1, dim2=2) == 0.0)


def test_spectral_function_sorted_and_conserves_weight():
    """Total spectral weight is E_tot^2 (Eq. 2.1 integrates to the square of
    total energy), and the support must come out ascending in omega."""
    e, c = _cloud(9, seed=14)
    w, wt = spectral_function(e, pairwise_omega(c))
    assert torch.all(w[:, 1:] >= w[:, :-1] - 1e-12)
    assert wt.sum(dim=1).item() == pytest.approx((e.sum(dim=1) ** 2).item(), rel=1e-12)


def test_cumulative_spectral_brackets_each_delta():
    e, c = _cloud(6, seed=15)
    _, wt = spectral_function(e, pairwise_omega(c))
    sm, sp = cumulative_spectral(wt)
    assert torch.allclose(sp - sm, wt)
    assert sm[:, 0].item() == pytest.approx(0.0, abs=1e-12)
    assert sp[:, -1].item() == pytest.approx(wt.sum(dim=1).item(), rel=1e-12)


def test_self_term_matches_two_point_correlator():
    """Eq. 2.11: SEMD against an empty (all energy at one point) reference
    reduces to sum_{i<j} 2 E_i E_j w_ij^2."""
    e, c = _cloud(8, seed=16)
    om = pairwise_omega(c)
    expected = (e.unsqueeze(2) * e.unsqueeze(1) * om ** 2).sum(dim=(1, 2))
    w, wt = spectral_function(e, om)
    assert (wt * w ** 2).sum(dim=1).item() == pytest.approx(expected.item(), rel=1e-12)


# --------------------------------------------------------------------------- #
# Energy balancing
# --------------------------------------------------------------------------- #
def test_extract_particles_conserves_energy():
    """Truncation must fold the dropped tail into ``residual``, never lose it."""
    img = torch.rand(4, 3, 16, 16, dtype=torch.float64)
    e, _, res = extract_particles(img, topk=10)
    assert torch.allclose(e.sum(dim=1) + res, img.sum(dim=(1, 2, 3)))


def test_extract_particles_threshold_moves_energy_to_residual():
    img = torch.rand(2, 1, 8, 8, dtype=torch.float64)
    e, _, res = extract_particles(img, topk=64, threshold=0.5)
    assert torch.all(e[e > 0] > 0.5)
    assert torch.allclose(e.sum(dim=1) + res, img.sum(dim=(1, 2, 3)))


def test_unequal_total_energy_is_handled_not_ignored():
    """Eq. 2.7 assumes equal E_tot; SR/HR violate that by construction. A pure
    rescale of the image must register as a nonzero distance rather than being
    silently normalized away."""
    img = torch.rand(2, 1, 12, 12, dtype=torch.float64) + 0.1
    same = semd_images(img, img, topk=32)
    scaled = semd_images(img * 1.5, img, topk=32)
    # Eq. 2.19 is a difference of three large, nearly-equal terms, so "zero" is
    # zero only to the cancellation floor — compare against the scale of the
    # terms being cancelled, not against an absolute constant.
    assert torch.all(same < 1e-6 * scaled)
    assert torch.all(scaled > 1e-8)


def test_balancing_is_symmetric_in_which_side_is_heavier():
    img_a = torch.rand(2, 1, 12, 12, dtype=torch.float64) + 0.1
    img_b = img_a * 1.4
    assert torch.allclose(
        semd_images(img_a, img_b, topk=32), semd_images(img_b, img_a, topk=32), rtol=1e-9
    )


def test_omega_R_scales_the_imbalance_penalty():
    """omega_R is an explicit knob, so it must demonstrably do something: a
    larger deposit distance costs more for the same energy deficit."""
    img = torch.rand(2, 1, 12, 12, dtype=torch.float64) + 0.1
    near = semd_images(img * 1.3, img, topk=32, omega_R=0.5)
    far = semd_images(img * 1.3, img, topk=32, omega_R=4.0)
    assert torch.all(far > near)


# --------------------------------------------------------------------------- #
# The discriminating test — finding (b), executable
# --------------------------------------------------------------------------- #
def test_geometry_discriminates_where_pixel_metrics_are_blind():
    """Two images with identical total energy, identical peak, and identical
    nonzero count, differing *only* in where the energy sits (1-prong vs
    2-prong). Every physics metric currently logged reports a perfect match;
    SEMD must not.

    This is the entire premise of adding SEMD, written as an assertion.
    """
    one_prong = torch.zeros(1, 1, 32, 32, dtype=torch.float64)
    two_prong = torch.zeros(1, 1, 32, 32, dtype=torch.float64)

    # Same four deposits, same energies, different spatial arrangement:
    # one tight cluster vs. two separated subjets.
    energies = [4.0, 3.0, 2.0, 1.0]
    tight = [(15, 15), (15, 16), (16, 15), (16, 16)]
    split = [(8, 8), (8, 9), (24, 24), (24, 25)]
    for e, (r, c) in zip(energies, tight):
        one_prong[0, 0, r, c] = e
    for e, (r, c) in zip(energies, split):
        two_prong[0, 0, r, c] = e

    # Every currently-logged physics metric sees a perfect match.
    assert energy_response(one_prong, two_prong).item() == pytest.approx(1.0, abs=1e-12)
    pk = peak_metrics(one_prong, two_prong)
    assert pk["peak_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert pk["nonzero_ratio"] == pytest.approx(1.0, abs=1e-12)

    # SEMD sees the difference.
    d = semd_images(one_prong, two_prong, topk=16)
    assert d.item() > 1.0, "SEMD must distinguish 1-prong from 2-prong geometry"


def test_pixel_permutation_changes_semd():
    """The direct statement of finding (b): shuffling pixels leaves every
    existing physics metric untouched but must move SEMD."""
    g = torch.Generator().manual_seed(42)
    img = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    img.view(-1)[torch.randperm(256, generator=g)[:20]] = torch.rand(
        20, generator=g, dtype=torch.float64
    ) + 0.5

    flat = img.reshape(1, -1)
    shuffled = flat[:, torch.randperm(flat.shape[1], generator=g)].reshape_as(img)

    # Permutation-invariant metrics: unchanged.
    assert energy_response(shuffled, img).item() == pytest.approx(1.0, abs=1e-12)
    assert peak_metrics(shuffled, img)["peak_ratio"] == pytest.approx(1.0, abs=1e-12)
    # SEMD: changed.
    assert semd_images(shuffled, img, topk=24).item() > 1e-6


def _two_deposit(sep: int) -> torch.Tensor:
    """Two deposits separated by ``sep`` columns — varies internal geometry
    without translating the jet."""
    img = torch.zeros(1, 1, 32, 32, dtype=torch.float64)
    img[0, 0, 16, 16] = 5.0
    img[0, 0, 16, 16 + sep] = 3.0
    return img


def test_closer_geometry_gives_smaller_distance():
    """Ordering sanity: a small change in prong separation must cost less than
    a large one. Note the comparison must vary *internal* geometry — a rigid
    shift of the whole jet is genuinely distance zero (see
    ``test_whole_image_translation_is_zero_distance``)."""
    base = _two_deposit(4)
    near = semd_images(_two_deposit(5), base, topk=8).item()
    far = semd_images(_two_deposit(12), base, topk=8).item()
    assert 0.0 < near < far


def test_whole_image_translation_is_zero_distance():
    """SEMD depends only on pairwise distances, so translating the entire jet
    leaves it exactly unchanged. This is a feature, not a limitation: a jet
    shifted within the image is the same jet, and the paper lists translation/
    rotation invariance among the properties that make SEMD a sensible
    collider observable."""
    base = _two_deposit(4)
    for shift in (1, 8):
        assert semd_images(base.roll(shifts=shift, dims=3), base, topk=8).item() == pytest.approx(
            0.0, abs=1e-9
        )


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_gradient_is_finite_through_the_sort():
    """The paper states every step including the sort is differentiable. If the
    gradient were NaN, ``--lambda-semd`` would poison training the moment it is
    switched on."""
    img = (torch.rand(2, 1, 16, 16, dtype=torch.float64) + 0.1).requires_grad_(True)
    target = torch.rand(2, 1, 16, 16, dtype=torch.float64) + 0.1

    loss = semd_images(img, target, topk=16).mean()
    loss.backward()

    assert img.grad is not None
    assert torch.isfinite(img.grad).all()
    assert img.grad.abs().sum() > 0.0


def test_gradient_finite_at_exact_identity():
    """The degenerate case: pred == target puts the sqrt at exactly zero, where
    an unclamped derivative would be infinite."""
    target = torch.rand(1, 1, 12, 12, dtype=torch.float64) + 0.1
    img = target.clone().requires_grad_(True)
    semd_images(img, target, topk=16).mean().backward()
    assert torch.isfinite(img.grad).all()


# --------------------------------------------------------------------------- #
# Batching / shape contracts
# --------------------------------------------------------------------------- #
def test_batch_independence():
    """Chunked evaluation must not mix samples together."""
    imgs = torch.rand(6, 2, 16, 16, dtype=torch.float64) + 0.1
    tgts = torch.rand(6, 2, 16, 16, dtype=torch.float64) + 0.1
    batched = semd_images(imgs, tgts, topk=16, chunk=4)
    singly = torch.cat([semd_images(imgs[i : i + 1], tgts[i : i + 1], topk=16) for i in range(6)])
    assert torch.allclose(batched, singly, rtol=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        semd_images(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16))


def test_multichannel_images_are_summed():
    """Channels are summed to one energy map, matching how the sample grids
    render a jet."""
    img = torch.rand(2, 3, 12, 12, dtype=torch.float64)
    summed = img.sum(dim=1, keepdim=True)
    assert torch.allclose(
        semd_images(img, img * 0 + 0.01, topk=16),
        semd_images(summed, summed * 0 + 0.03, topk=16),
        rtol=1e-9,
    )


def test_periodic_phi_wraps():
    """With periodic phi, two deposits straddling the column seam are close
    together, not a full image-width apart.

    A pair is required: a single deposit has no pairwise structure, so any two
    one-particle images are at distance zero however phi is treated.
    """
    seam = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    seam[0, 0, 4, 0] = 5.0
    seam[0, 0, 4, 7] = 3.0  # 1 apart when wrapped, 7 apart when flat

    ref = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    ref[0, 0, 4, 3] = 5.0
    ref[0, 0, 4, 4] = 3.0  # 1 apart either way

    wrapped = semd_images(seam, ref, topk=4, periodic_phi=True)
    flat = semd_images(seam, ref, topk=4, periodic_phi=False)
    assert wrapped.item() == pytest.approx(0.0, abs=1e-9)
    assert flat.item() > 1.0


# --------------------------------------------------------------------------- #
# Loss scaling — the trap that normalization exists to close
# --------------------------------------------------------------------------- #
def test_semd_loss_normalization_brings_it_onto_l1_scale():
    """Raw SEMD carries units of energy^2 * length^2 and measures ~1e8 on real
    calorimeter data, against ~2 for the L1 term. Without normalization an
    innocuous ``--lambda-semd 0.01`` makes SEMD >99.99% of the generator loss,
    silently replacing the objective. Normalizing by E_tot_HR^2 — the natural
    scale of the spectral function, since Eq. 2.1 integrates to E_tot^2 — must
    put it within a few orders of magnitude of the other terms.
    """
    from multiscale_sr.engine import semd_loss

    torch.manual_seed(0)
    # Energies of a realistic magnitude, not O(1).
    pred = torch.rand(2, 3, 32, 32, dtype=torch.float64) * 50.0
    target = torch.rand(2, 3, 32, 32, dtype=torch.float64) * 50.0

    raw = semd_loss(pred, target, topk=32, normalize_by_energy=False)
    normed = semd_loss(pred, target, topk=32, normalize_by_energy=True)

    assert normed < raw
    # The normalized value must be small enough that lambda ~ O(1) is sane.
    assert normed.item() < 1e3, f"normalized SEMD still huge: {normed.item()}"


def test_semd_loss_normalization_preserves_ranking():
    """Normalization must rescale, not reorder: a worse geometric match has to
    stay worse. If it did not, the loss and the reported metric could disagree
    about which of two models is better."""
    from multiscale_sr.engine import semd_loss

    target = torch.zeros(1, 1, 32, 32, dtype=torch.float64)
    target[0, 0, 16, 16] = 5.0
    target[0, 0, 16, 20] = 3.0

    near, far = _two_deposit(5), _two_deposit(12)
    for norm in (True, False):
        a = semd_loss(near, target, topk=8, normalize_by_energy=norm)
        b = semd_loss(far, target, topk=8, normalize_by_energy=norm)
        assert a < b, f"ranking broken with normalize_by_energy={norm}"


def test_semd_loss_gradient_finite_when_normalized():
    from multiscale_sr.engine import semd_loss

    pred = (torch.rand(2, 1, 16, 16, dtype=torch.float64) * 20.0).requires_grad_(True)
    target = torch.rand(2, 1, 16, 16, dtype=torch.float64) * 20.0
    semd_loss(pred, target, topk=16).backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0.0
