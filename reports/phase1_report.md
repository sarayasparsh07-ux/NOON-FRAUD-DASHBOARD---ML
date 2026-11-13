# Phase 1 Report — GAN-Augmented Fraud Detection, Built Blind

> *Archived Phase 1 submission report (13 Oct 2026), kept unchanged as a supporting document. It was written before Noon's materials were released, so its "pending" statements describe that date. The integrated report for the whole project is `final_report.md`.*

**Sparsh Saraya · Noon ML internship challenge · Phase 1 submission (Phases 2–4 pending Noon's materials)**

**What I assumed fraud looks like.** From the 11 column names alone, I modelled UAE food-delivery fraud as a mixture of five typologies, each with an innocent look-alike: promo abuse (brand-new account, discount, small basket), refund abuse (several prior refunds), stolen cards (new account, card, large basket, location mismatch, night uplift), account takeover (*old* account, new device, mismatch, order burst) and COD fake orders. Anchors came from public research (UAE basket ≈ AED 77; 1–2 orders a week; ~3% food-delivery block rate). Every magnitude is an assumption, listed in `assumptions_register.md`; how right I was can only be judged on real data in Phase 3.

**Seed data.** 50,000 orders at an illustrative 3% fraud rate from a latent-profile mixture with label noise (3% of fraud recorded as legitimate). No feature gives fraud away (best single-feature AUC 0.76). Split 70/15/15 before anything was fitted; the test set holds 227 frauds.

**Did the GAN really train?** Yes. CTGAN (ctgan 0.12.1) was fitted on the 1,058 training-partition fraud rows only (verified by content hash) for 600 epochs: generator loss fell from +0.9 to −2.0 and flattened while discriminator loss stayed near 0; fidelity improved from 100 to 600 epochs; a seeded refit reproduces identical losses and rows. It kept joint structure that a column-by-column sampler destroys (account-takeover signature 4.4% vs 3.7% real; ablation 1.1%) and copied no rows. Its main flaw: it smears the brand-new-account mode of `account_age_days` (31% of rows at 0 days vs 14% real), which keeps its rows detectable (two-sample AUC 0.78 vs 0.53 real-vs-real).

**Did GAN augmentation improve detection? No — not measurably.** Same gradient-boosting model, thresholds tuned on validation, one untouched test set:

| Training arm | Recall | Precision | F1 | AP (PR-AUC) | False positives | Frauds caught (of 227) |
|---|---|---|---|---|---|---|
| A Baseline (**selected**) | 51.1% | 54.5% | 0.527 | 0.516 | 97 | 116 |
| B + CTGAN fraud (0.25×) | 48.9% | 57.8% | 0.530 | 0.510 | 81 | 111 |
| C Class-weighted | 47.6% | 52.2% | 0.498 | 0.454 | 99 | 108 |
| D + SMOTE-NC | 51.5% | 51.3% | 0.514 | 0.517 | 111 | 117 |
| E + duplicated fraud | 57.7% | 49.1% | 0.530 | 0.525 | 136 | 131 |

CTGAN − Baseline: AP −0.007 [95% CI −0.020, +0.012], F1 +0.003 [−0.023, +0.026], false positives −16 [−27, −5]. Across five full re-runs CTGAN had fewer false positives *and* lower recall in 5/5 seeds, and the same held at 1% and 5% prevalence. More CTGAN rows made validation AP fall (0.521 at 0.25× → 0.457 at 2×). The GAN shifted the operating point towards fewer false alarms rather than catching more fraud, so the pre-declared rule selected the simpler Baseline. Class weighting clearly hurt (AP −0.062); duplication bought recall with 39 extra false alarms; at the default 0.5 threshold the comparison would have been misleading.

**Stress test.** Under deliberately different assumptions every model degraded alike: population drift raised false alarms by ~47% at the frozen threshold, adaptive fraud halved AP (0.55 → 0.24), and an unseen typology was caught 11% of the time. The GAN added no robustness — it can only re-express the fraud it was shown.

**Why this is not proof of real-world performance.** The classifier, the GAN and the test set all come from rules I wrote, so high scores show that the pipeline recovers planted patterns, not Noon's fraud. Only Phase 3's real data can judge that.

**Why not just ask an LLM to "make more fraud rows"?** An LLM writes rows from its general priors about what fraud "sounds like": it never fits the data, gives no guarantee that frequencies and column relationships match, and cannot be tested as a sample from a distribution. A GAN *learns* the joint distribution of the actual fraud examples by pitting a generator against a discriminator, so its output can be measured (losses, fidelity reports, two-sample tests), reproduced from a seed, conditioned, and trained inside Noon's environment on confidential data no prompt could contain. The flip side is this phase's key lesson: my GAN learned from my *guesses*, so it could only re-express them. A GAN trained on Noon's real fraud carries patterns nobody wrote down — Noon's advantage, to be measured in Phase 3.

*Pending (Phases 2–3): comparison with Noon's GAN data and classifier on real transactions, and the explanation of the gap.*
