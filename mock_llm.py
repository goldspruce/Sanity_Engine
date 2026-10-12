"""Offline stand-in (no API). Parses real passages from the prompt so quotes are verbatim; plants one fake quote and one ungrounded number."""
import json, re

def _passages(user):
    blob = user.split("PASSAGES:\n", 1)[1]
    return json.loads(blob)

def _first_sentence(t):
    return re.split(r"(?<=[.!?])\s", t.strip())[0 if not t.startswith("DEMO") else 1].strip() if "." in t else t

class MockLLM:
    def json(self, system, user, schema, temperature=0.0):
        n = schema.__name__
        if n == "AttackPlan":
            return dict(
                claims=[dict(id="C1", text="Focused 100-120 kHz ultrasound delivers usable power across 3 m of air"),
                        dict(id="C2", text="A PZT receiver converts it to 5 W DC efficiently"),
                        dict(id="C3", text="Inaudible therefore safe")],
                queries=[dict(text="why power transfer across air-ceramic boundary is inefficient", kind="impossibility"),
                         dict(text="absorption of ultrasound in air at long range", kind="limit"),
                         dict(text="limits of focusing acoustic beams diffraction aperture", kind="limit"),
                         dict(text="exposure limits airborne ultrasound standards", kind="failure_mode")],
                recall=[dict(constraint="Impedance mismatch reflects nearly all power at air/ceramic boundary",
                             book="(mock book A)", search_terms="acoustic impedance mismatch reflected transmitted power"),
                        dict(constraint="Nonlinear harmonic distortion in high-intensity piezo stacks limits output",
                             book="(mock book B)", search_terms="harmonic distortion piezoelectric stack saturation nonlinear")])
        if n == "Review":
            ps = _passages(user)
            def find(kw):
                return next((p for p in ps if kw in p["text"].lower()), None)
            imp, ab, safe, foc = find("impedances"), find("absorption in air"), find("inaudible"), find("focusing")
            def sent(p, kw):
                body = [s for s in re.split(r"(?<=[.!?])\s", p["text"].replace("\n", " ")) if len(s.split()) >= 6]
                return next((s for s in body if kw in s.lower()), body[0]).strip()
            issues = []
            if imp: issues.append(dict(id="I1", severity="critical", category="Boundary mechanics", claim_id="C2",
                claim="The air-to-ceramic boundary reflects almost all incident power.", passage_id=imp["id"],
                quote=sent(imp, "impedances"), edge_condition="Would need coupling that removes the impedance mismatch."))
            if ab: issues.append(dict(id="I2", severity="major", category="Propagation", claim_id="C1",
                claim="Air absorption at ultrasonic frequency costs about 160 dB over the path.", passage_id=ab["id"],
                quote=sent(ab, "absorption"), edge_condition="Lower frequency or shorter path."))
            if safe: issues.append(dict(id="I3", severity="major", category="Safety", claim_id="C3",
                claim="Inaudibility is not a safety argument.", passage_id=safe["id"],
                quote=sent(safe, "inaudible"), edge_condition="Exposure compliance shown against a cited standard."))
            if foc: issues.append(dict(id="I4", severity="critical", category="Focusing", claim_id="C1",
                claim="Long-range focus needs a very large aperture.", passage_id=foc["id"],
                quote="The aperture must always exceed fifty wavelengths for any focal range.",  # planted fabricated quote
                edge_condition="Array aperture large versus wavelength."))
            return dict(
                scientific_synthesis="While ultrasonic acoustic focusing across 3 m is theoretically feasible, the steep impedance mismatch between air and piezoceramics reflects >99.99% of energy, and atmospheric attenuation at 120 kHz severely degrades transmission.",
                falsification_test="Direct calorimetric or electrical power measurement across an air-ceramic interface at 120 kHz over a 3-meter path.",
                issues=issues,
                untested_claims=["Rectification efficiency of the PZT receiver (no passage tests it)"]
            )
        if n == "Entail":
            return dict(label="supports", why="mock")
