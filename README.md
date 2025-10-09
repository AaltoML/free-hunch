---
layout: default
title: Free Hunch — Denoiser Covariance Estimation for Diffusion Models Without Extra Costs
permalink: /
---

<div align="center">

# <strong>Free Hunch</strong>: Denoiser Covariance Estimation for Diffusion Models Without Extra Costs

<p>
  <a href="mailto:severi.rissanen@aalto.fi">Severi Rissanen</a> ·
  <a href="https://users.aalto.fi/~heinonem/">Markus Heinonen</a> ·
  <a href="https://users.aalto.fi/~asolin/">Arno Solin</a><br/>
  Aalto University
</p>

<h2>ICLR 2025</h2>

<h3>
  <a href="https://arxiv.org/abs/2410.11149">Paper (arXiv)</a> |
  <a href="https://github.com/AaltoML/free-hunch">Code</a> |
  <a href="#bibtex">BibTeX</a>
</h3>

</div>

---

<div align="center">
  <img src="{{ '/assets/free-hunch/fig_1.png' | relative_url }}" alt="Deblurring comparison across methods with few solver steps" style="max-width:100%; border-radius:12px;" />
  <p><em>Teaser.</em> With few sampler steps, accurate denoiser covariance turns out to be crucial for high-fidelity reconstruction.</p>
</div>

## TL;DR

- **Covariance for free.** We reuse information already present in training data and the generative trajectory to estimate the denoiser covariance—no retraining, no score Hessians.  
- **Two lightweight updates.** A **time update** transfers covariance across noise levels; a **space update** performs BFGS-like low-rank corrections along the sampler path.  
- **Plug-and-play guidance.** Better covariance → well-scaled, stable reconstruction guidance → sharper details at low step counts.

## Abstract

The conditional score for inverse problems needs the denoiser **mean and covariance** of \(p(x_0 \mid x_t)\). Prior work either adds heavy test-time compute, modifies training/architecture, or uses crude (often diagonal) covariances. **Free Hunch (FH)** integrates two *free* sources: (i) data covariance (DCT-diagonal for images) and (ii) curvature observed along the generative trajectory via a BFGS-style online update. A simple time-transfer rule moves covariance between noise levels. On ImageNet inverse problems (deblurring, inpainting, super-resolution), FH improves quality—especially LPIPS—at small step counts, while staying training-free and architecture-agnostic.

## Method at a glance

**Tweedie link (2nd order).** The Tweedie identities connect the score to the denoiser mean and covariance; the covariance involves the Hessian of \(\log p(x_t)\). Instead of computing Hessians, FH approximates the covariance via:

1. **Time update.** Transfer \(\Sigma_{0\mid t}(x_t)\) → \(\Sigma_{0\mid t+\Delta t}(x_t)\) analytically using a local Gaussian approximation of \(p(x_t)\) and the forward SDE evolution.
2. **Space update.** When the sampler moves from \(x_t\) to \(x_t+\Delta x\) at the *same* time \(t\), use a BFGS-like low-rank correction based on finite differences of the denoiser mean \(\mu_{0\mid t}\).
3. **Efficient representation.** Maintain \(\Sigma\) as **D + U Uᵀ − V Vᵀ** so both updates and inverses stay cheap (Woodbury on small \(k \times k\) systems).
4. **Initialization.** For images, start from **DCT-diagonal** data covariance; it’s a strong prior and avoids early over/under-scaling.

<div align="center">
  <img src="{{ '/assets/free-hunch/fig_2.png' | relative_url }}" alt="Posterior geometry and effect of covariance choice" style="max-width:100%; border-radius:12px;" />
  <p><em>Geometry.</em> Poor (diagonal) covariances distort guidance geometry; FH aligns the local posterior shape with the sampler’s trajectory.</p>
</div>

## Why it helps (guidance scale)

Diagonal/identity covariances can **over-amplify** the conditional term, especially at high noise levels and in high dimensions—forcing post-hoc clipping or ad-hoc scaling. With FH, the guidance magnitude is naturally calibrated, reducing (or removing) the need for such tricks.

<div align="center">
  <img src="{{ '/assets/free-hunch/fig_guidance_strength.png' | relative_url }}" alt="LPIPS vs guidance strength; FH needs little or no scaling" style="max-width:420px; border-radius:12px;" />
  <p><em>Less tuning, more fidelity.</em> With better covariance, the optimal guidance is close to 1 (no scaling).</p>
</div>

## Results snapshot

Across four linear inverse problems on ImageNet 256×256—Gaussian/motion deblurring, random inpainting, and 4× super-resolution—FH (and FH+Online) consistently improves perceptual quality at **low step counts** (e.g., 15/30-step Heun), with strong LPIPS and crisp details. See the paper for full tables, ablations, and more qualitative results.

## Get the code

- Repository: <a href="https://github.com/AaltoML/free-hunch">github.com/AaltoML/free-hunch</a>  
- Typical setup (see repo for exact instructions):
  ```bash
  git clone https://github.com/AaltoML/free-hunch.git
  cd free-hunch
  # (optional) conda create -n free-hunch python=3.10 && conda activate free-hunch
  pip install -e .

