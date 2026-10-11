"""Owner V. The per-area question for the vision model: instructions, user text and answer schema.

Bump PROMPT_VERSION on every change to SYSTEM, area_prompt or answer_schema: it is part of the cache key.
"""
from pipeline.common.config import STOCK_VALUES

PROMPT_VERSION = "v2"
PROMPT_VERSION_REF = "v3"  # exemplars + evidence fields + "lower when unsure"

# Values of `observed` besides the candidate product ids.
NO_PRODUCT = ("other", "empty", "occluded")

SYSTEM = """You inspect one shelf area of a supermarket fruit and vegetable section, seen from a ceiling camera, for store staff.

You get two pictures of the same area:
1. Context view: the camera picture around the area. The area is the region inside the thin magenta outline. Everything outside the outline belongs to neighbouring areas and must not be counted. Flat grey regions are outside the camera's view.
2. Flattened view: only the inside of the outline, stretched to a square. Its proportions are distorted. Black regions are outside the camera's view.

Judge only what is inside the outline, and only the part the camera can see.

Stock level of the area:
- empty: 0% to 10% of the area is covered or piled; bare shelf, tray, crate or basket, or only a few stray items.
- half: 10% to 50% of the area is covered or piled.
- full: more than 50% of the area is covered or piled.
Loose produce: judge coverage and pile height. Packaged produce (bags, nets, punnets): judge how many packs there are against the space. Price labels, shelf rails, crates, baskets, plastic liners and paper are not product. Give your honest best estimate; do not lean low or high.

Product in the area:
- observed is the candidate that fills most of the area. Use "other" if it is a product that is not in the candidate list, and "empty" if you can see the bare shelf and there is no product to identify. Use "occluded" if a person, a trolley, a night blind (a grey fabric screen pulled down over the shelves) or anything else hides the area so that you cannot see the shelf; then give your best guess for stock and a low confidence.
- matches_expected is true when the product in the area is the expected product, and false when a different product fills most of it. When observed is "empty" or "occluded" there is no evidence of a wrong product: answer true.
- Varieties of the same fruit can look alike from the ceiling. Use the hints, and say false only when you can see a real difference.

Write reason first: one short sentence on what you see inside the outline. confidence is your probability, from 0 to 1, that both stock and matches_expected are right."""


SYSTEM_REF = SYSTEM.replace(
    "Give your honest best estimate; do not lean low or high.",
    "When you are unsure between two levels, answer the LOWER one: a wrongly high answer costs more than a wrongly low one.",
) + """

Evidence first: before the verdict give floor_visible_pct, the share of the area (0 to 100) where bare crate, tray or shelf floor shows between or instead of product, and view: clear, dark (too dark or washed out to judge) or blocked (a person, trolley or night blind hides it).

Reference pictures: after the two views you may get flattened views of this same area from earlier today, each labelled by a human with its stock level. Compare the area now with them; the same crate can hold a different amount, and the labels are the store's own scale for full, half and empty."""


def describe(product_id: str, name_pt: str, products: dict) -> str:
    """One line for a product: id, English name, unit and visual hint (Portuguese name only when it is unknown)."""
    info = products.get(product_id)
    if info is None:
        return f"{product_id}: {name_pt}"
    return f"{product_id}: {info['name_en']} [{info['unit']}] - {info['hint']}"


def area_prompt(expected_id: str, candidates: dict[str, str], products: dict, exemplar_levels: tuple = ()) -> str:
    """User text for one area. candidates is {product_id: name_pt} for every product on that camera.
    exemplar_levels: stock levels of the labelled reference pictures that follow the two views (3rd picture on)."""
    ref = [f"Pictures 3 to {2 + len(exemplar_levels)} are earlier flattened views of this same area, labelled by a human: "
           + ", ".join(f"picture {3 + i} = {lv}" for i, lv in enumerate(exemplar_levels)) + "."] if exemplar_levels else []
    lines = [
        "Expected product in this area (from the planogram):",
        describe(expected_id, candidates[expected_id], products),
        "",
        "Candidate products on this camera (answer observed with one of these ids):",
        *(describe(pid, name, products) for pid, name in sorted(candidates.items())),
        "",
        "Picture 1 is the context view, picture 2 is the flattened view. Answer for the area inside the magenta outline.",
        *ref,
    ]
    return "\n".join(lines)


def answer_schema(candidate_ids: list[str], evidence: bool = False) -> dict:
    """JSON schema of the answer (the shape of one provider entry in vlm.json, see CONTRACTS.md).
    evidence: also ask floor_visible_pct and view, before the verdict (prompt v3)."""
    properties = {
        "reason": {"type": "string"},
        **({"floor_visible_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            "view": {"type": "string", "enum": ["clear", "dark", "blocked"]}} if evidence else {}),
        "observed": {"type": "string", "enum": sorted(candidate_ids) + list(NO_PRODUCT)},
        "matches_expected": {"type": "boolean"},
        "fill_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "stock": {"type": "string", "enum": list(STOCK_VALUES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {"type": "object", "properties": properties, "required": list(properties), "propertyOrdering": list(properties)}


def clean_answer(answer: dict, candidate_ids: list[str]) -> dict:
    """Check and normalise a parsed model answer. Raises ValueError when it does not follow the schema."""
    if answer.get("stock") not in STOCK_VALUES:
        raise ValueError(f"bad stock: {answer.get('stock')!r}")
    if answer.get("observed") not in (*candidate_ids, *NO_PRODUCT):
        raise ValueError(f"bad observed: {answer.get('observed')!r}")
    if not isinstance(answer.get("matches_expected"), bool):
        raise ValueError(f"bad matches_expected: {answer.get('matches_expected')!r}")
    extra = {k: answer[k] for k in ("floor_visible_pct", "view") if k in answer}
    return {
        **extra,
        "stock": answer["stock"],
        "fill_pct": min(100, max(0, int(answer["fill_pct"]))),
        "matches_expected": answer["matches_expected"],
        "observed": answer["observed"],
        "confidence": min(1.0, max(0.0, float(answer["confidence"]))),
        "reason": str(answer.get("reason", "")),
    }
