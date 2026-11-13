"""Render reports/assumptions_register.md from the template + the live generator parameters."""
import json
from .config import REPORTS
from .seed_generator import params_to_markdown, STRESS_OVERRIDES

DESCRIPTIONS = {
    "S0_in_distribution": "Fresh draw from the unchanged training generator (reference).",
    "S1_population_drift": "Legitimate customers change; fraud does not.",
    "S2_adaptive_fraud": "Fraudsters adapt to look more legitimate.",
    "S3_unseen_typology": "A new fraud typology (wallet laundering) absent from training forms 30% of fraud.",
}


def main():
    tpl = (REPORTS / "_assumptions_register_template.md").read_text()
    stress = []
    for name, ov in STRESS_OVERRIDES.items():
        stress.append(f"- **{name}** — {DESCRIPTIONS[name]}")
        if ov:
            stress.append("  ```json\n  " + json.dumps(ov, default=str).replace("\n", "\n  ") + "\n  ```")
    out = tpl.replace("{{PARAMS}}", params_to_markdown()).replace("{{STRESS}}", "\n".join(stress))
    (REPORTS / "assumptions_register.md").write_text(out)
    print("wrote", REPORTS / "assumptions_register.md")


if __name__ == "__main__":
    main()
