"""Pre-flight parameter validation, with no get_data call (SPEC.md §5.5).

Checks, in order:
  1. the view exists;
  2. no parameter outside the view's list;
  3. every mandatory parameter is present;
  4. each value passes rpdata's `Parameter.validate` (normalising when exactly one
     safe candidate exists);
  5. cross-parameter rules: `rpdata.validate_params` when it exists (R1), otherwise
     the rules learned in the notes.
Then each item is marked certain or uncertain (SPEC.md §5.3, §5.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import rpdata

from ..memory.knowledge import KnowledgeBase
from ..memory.learning import DOMAIN_CLOSED, NORMALISED, VALUES_VALIDATED, LearningEvent
from ..memory.values import CLOSED_UNPUBLISHED, LARGE_CLOSED, SMALL_CLOSED, ValueDomains
from .canonical import CanonicalRequest
from .catalogue import Catalogue

OK = "ok"
UNCERTAIN = "uncertain"
MISSING = "missing"

_YMD_RE = re.compile(r"^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DMY_OR_MDY_RE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")


@dataclass
class ValidationResult:
    params: dict[str, Any] = field(default_factory=dict)
    status: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    events: list[LearningEvent] = field(default_factory=list)
    note_ids: dict[str, list[str]] = field(default_factory=dict)

    @property
    def all_certain(self) -> bool:
        return not self.issues and all(s["status"] == OK for s in self.status.values())


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


class ParamValidator:
    def __init__(self, catalogue: Catalogue, knowledge: KnowledgeBase, domains: ValueDomains,
                 settings) -> None:
        self.catalogue = catalogue
        self.kb = knowledge
        self.domains = domains
        self.settings = settings

    # ------------------------------------------------------------- helpers

    def is_date_param(self, name: str) -> bool:
        fmt = self.catalogue.params[name].format
        return bool(fmt and fmt.startswith("yyyy-mm-dd"))

    def options_for(self, name: str, request: str, value: Any) -> list[str]:
        """Options shown in a review, sized by the parameter's value domain (§5.6, §8.4)."""
        domain = self.domains.domain(name)
        if domain == SMALL_CLOSED:
            return self.domains.allowed_values(name)
        if domain in (LARGE_CLOSED, CLOSED_UNPUBLISHED):
            extra = (str(value),) if isinstance(value, str) else ()
            return self.domains.shortlist(name, request, self.settings.review_options_size,
                                          extra_terms=extra)
        return []

    def _normalise_date(self, value: str) -> tuple[str | None, list[str]]:
        """(certain ISO form, or None; ambiguous candidates)."""
        if m := _YMD_RE.match(value) or _COMPACT_RE.match(value):
            iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return iso, []
        if m := _DMY_OR_MDY_RE.match(value):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dmy, mdy = _iso(y, b, a), _iso(y, a, b)
            convention = self.kb.date_convention()
            if convention == "dmy" and dmy:
                return dmy, []
            if convention == "mdy" and mdy:
                return mdy, []
            candidates = sorted({c for c in (dmy, mdy) if c})
            if len(candidates) == 1:
                return candidates[0], []
            return None, candidates
        return None, []

    def _variants(self, value: str) -> list[str]:
        no_space = re.sub(r"\s+", "", value)
        forms = [value.strip(), value.upper(), value.lower(), value.title(), no_space,
                 no_space.upper(), no_space.lower(), re.sub(r"\s+", "_", value.strip()).upper()]
        return [v for v in dict.fromkeys(forms) if v != value]

    def _passes(self, view: str, name: str, value: Any) -> bool:
        try:
            self.catalogue.validate_param(view, name, value)
            return True
        except rpdata.RPDataError:
            return False

    # --------------------------------------------------------------- check

    def check(
        self,
        view: str,
        proposal: dict[str, dict[str, Any]],
        request: str,
        approved: set[str],
        rules_overridden: bool,
        failed_keys: list[str],
    ) -> ValidationResult:
        res = ValidationResult()
        info = self.catalogue.views[view]  # step 1 is done by select_view
        validated_values: dict[str, Any] = {}

        # Step 2: no parameter outside the view's list.
        for name in proposal:
            if name not in info.all_params:
                res.adjustments.append(f"dropped {name} (not a parameter of {view})")

        for name in info.all_params:
            prop = proposal.get(name)
            value = None if prop is None else prop.get("value")
            mandatory = name in info.mandatory_params
            if value is None:
                # Step 3: mandatory parameters.
                if mandatory:
                    res.status[name] = {
                        "status": MISSING, "value": None,
                        "message": "needed",
                        "options": self.options_for(name, request, None),
                    }
                continue

            note_ids = list(prop.get("note_ids", []))
            entry: dict[str, Any] = {"status": OK, "value": value, "message": "", "options": []}
            final = value

            # Dates: an unambiguous reformat is certain (rpdata's format is yyyy-mm-dd).
            if self.is_date_param(name) and isinstance(value, str) and not self._passes(
                view, name, value
            ):
                iso, candidates = self._normalise_date(value)
                if iso and self._passes(view, name, iso):
                    final = iso
                    res.adjustments.append(f"{name} {value!r} → {iso!r} (date format)")
                    res.events.append(LearningEvent.make(
                        NORMALISED, param=name, view=view, reason="date_format",
                        message="dates must be yyyy-mm-dd strings", **{"from": value, "to": iso}))
                elif candidates:
                    entry.update(status=UNCERTAIN, options=candidates,
                                 message=f"{value!r} is ambiguous (day/month order)")

            # Step 4: rpdata's own validator.
            if entry["status"] == OK:
                try:
                    self.catalogue.validate_param(view, name, final)
                    validated_values[name] = final
                except rpdata.RPDataError as exc:
                    final = self._try_normalise(view, name, final, exc, res, entry, request)

            # Certainty (§5.4): approved by a human, or high-confidence with a real source.
            if entry["status"] == OK and name not in approved:
                alias_ids = self._alias_support(name, final, request)
                if alias_ids:
                    note_ids = sorted(set(note_ids) | set(alias_ids))
                elif prop.get("source") == "guess" or prop.get("confidence") != "high":
                    entry.update(
                        status=UNCERTAIN,
                        message="guessed" if prop.get("source") == "guess" else "inferred",
                    )
                    entry["options"] = self.options_for(name, request, final)

            # Coverage notes: a date the notes place outside the data period (§5.4).
            if (entry["status"] == OK and name not in approved and self.is_date_param(name)
                    and (warning := self._coverage_warning(final))):
                entry.update(status=UNCERTAIN, message=warning)

            entry["value"] = final
            res.status[name] = entry
            res.params[name] = final
            res.note_ids[name] = note_ids

        # Step 5: cross-parameter rules.
        self._cross_parameter_checks(view, res, rules_overridden)

        # A call that already failed in this request is never repeated (§6.3).
        key = CanonicalRequest.build(view, res.params).key()
        if key in failed_keys:
            res.issues.append({
                "key": "already_failed", "overridable": False, "params": [],
                "message": "This exact query already failed in this request; change it "
                           "or cancel.",
                "suggestions": [],
            })

        string_values = {k: v for k, v in validated_values.items() if isinstance(v, str)}
        if string_values:
            res.events.append(LearningEvent.make(VALUES_VALIDATED, values=string_values))
        return res

    def _try_normalise(self, view, name, value, exc, res, entry, request) -> Any:
        """Step 4 failed: look for exactly one safe candidate (§5.5)."""
        if isinstance(exc, rpdata.UnknownNodeError) or "not a known" in str(exc):
            res.events.append(LearningEvent.make(
                DOMAIN_CLOSED, param=name, message=str(exc).split(": ", 1)[-1]))
        candidates: set[str] = set()
        if isinstance(value, str):
            candidates.update(self.domains.normalisation_candidates(name, value))
            candidates.update(v for v in self._variants(value) if self._passes(view, name, v))
            candidates = {c for c in candidates if self._passes(view, name, c)}
        if len(candidates) == 1:
            fixed = candidates.pop()
            reason = "case" if fixed.upper() == str(value).upper() else "whitespace"
            res.adjustments.append(f"{name} {value!r} → {fixed!r} ({reason})")
            res.events.append(LearningEvent.make(
                NORMALISED, param=name, view=view, reason=reason,
                message=str(exc).split(": ", 1)[-1], **{"from": value, "to": fixed}))
            return fixed
        message = str(exc).split(": ", 1)[-1]
        options = sorted(candidates) or self.options_for(name, request, value)
        entry.update(status=UNCERTAIN, message=message, options=options)
        return value

    def _alias_support(self, name: str, value: Any, request: str) -> list[str]:
        """IDs of likely/certain aliases in the request that map to this value."""
        return [
            a["id"] for a in self.kb.aliases_in_text(name, request)
            if a.get("value") == value and a.get("certainty") in ("likely", "certain")
        ]

    def _coverage_warning(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cov = self.kb.coverage()
        if cov.get("empty_after") and cov.get("last_seen") and value >= cov["empty_after"]:
            return (f"data I have seen ends {cov['last_seen']}; dates from "
                    f"{cov['empty_after']} returned no data before")
        if cov.get("empty_before") and cov.get("first_seen") and value <= cov["empty_before"]:
            return (f"data I have seen starts {cov['first_seen']}; dates up to "
                    f"{cov['empty_before']} returned no data before")
        try:
            weekday = date.fromisoformat(value).weekday()
        except ValueError:
            return None
        weekend_rule = [r for r in self.kb.rules("general", "general", "coverage")
                        if r.get("dedup_key") == "coverage:weekend"
                        and r["certainty"] in ("likely", "certain")]
        if weekend_rule and weekday >= 5:
            return "this is a weekend; weekend dates have returned no data"
        return None

    def _cross_parameter_checks(self, view: str, res: ValidationResult,
                                rules_overridden: bool) -> None:
        if any(s["status"] != OK for s in res.status.values()):
            return  # fix single-parameter problems first

        native = self.catalogue.native_validate_params()
        if native is not None:
            try:
                native(view, **res.params)
            except rpdata.RPDataError as exc:
                res.issues.append({"key": "rpdata", "overridable": False, "params": [],
                                   "message": str(exc), "suggestions": []})
            return

        if rules_overridden:
            return
        for _, _, rule in self.kb.all_rules("order"):
            low, high = rule.get("data", {}).get("low"), rule.get("data", {}).get("high")
            if low in res.params and high in res.params and res.params[low] > res.params[high]:
                res.issues.append({
                    "key": f"rule:{rule['id']}", "overridable": True, "params": [low, high],
                    "message": f"{low} must not be after {high} ({rule['text']})",
                    "suggestions": [],
                })
        for rule in self.kb.rules("view", view, "requires_filter"):
            data = rule.get("data", {})
            param, value = data.get("param"), data.get("value")
            if res.params.get(param) != value:
                continue
            insufficient = data.get("insufficient", {})
            others = {k: v for k, v in res.params.items() if k != param}
            if all(k in insufficient and insufficient[k] in ("*", v) for k, v in others.items()):
                info = self.catalogue.views[view]
                suggestions = [p for p in info.optional_params
                               if p not in res.params and p not in insufficient]
                res.issues.append({
                    "key": f"rule:{rule['id']}", "overridable": True, "params": [param],
                    "message": f"This would time out ({rule['text']}). Add a filter.",
                    "suggestions": suggestions,
                })
