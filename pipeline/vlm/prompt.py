"""Owner V. The question for the vision model: instructions, message parts and answer schema.

One call covers several areas of one frame (usually all of them): the candidate list is sent once, then
each area as a text line followed by its context crop. Bump PROMPT_VERSION on every change to SYSTEM,
frame_parts or frame_schema: it is part of the cache key.
"""
from pipeline.common.config import STOCK_VALUES

PROMPT_VERSION = "v3"

# Values of `observed` besides the candidate product ids.
NO_PRODUCT = ("other", "empty", "occluded")

SYSTEM = """You inspect shelf areas of a supermarket fruit and vegetable section, seen from a ceiling camera, for store staff.

The message lists the candidate products of this camera, then one or more areas. Each area starts with a line "Area <id>" naming its expected product, followed by its picture: the camera picture around the area, with the area inside a thin magenta outline. Everything outside the outline belongs to neighbouring areas and must not be counted. Flat grey regions are outside the camera's view. Judge each area on its own: neighbouring areas can hold other products and other stock levels.

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

Answer one entry per area, in the order given. In each entry write reason first: one short sentence on what you see inside the outline. confidence is your probability, from 0 to 1, that both stock and matches_expected are right."""


def describe(product_id: str, name_pt: str, products: dict) -> str:
    """One line for a product: id, English name, unit and visual hint (Portuguese name only when it is unknown)."""
    info = products.get(product_id)
    if info is None:
        return f"{product_id}: {name_pt}"
    return f"{product_id}: {info['name_en']} [{info['unit']}] - {info['hint']}"


def frame_parts(areas: list[tuple[str, str, bytes]], candidates: dict[str, str], products: dict) -> list[str | bytes]:
    """Message parts for one call: text and PNG bytes, interleaved so each picture follows its area line.

    areas is [(area_id, expected product_id, context crop PNG)]; candidates is {product_id: name_pt} for the camera.
    """
    header = "\n".join([
        "Candidate products on this camera (answer observed with one of these ids):",
        *(describe(pid, name, products) for pid, name in sorted(candidates.items())),
    ])
    parts: list[str | bytes] = [header]
    for area_id, expected_id, image in areas:
        name = products.get(expected_id, {}).get("name_en", candidates[expected_id])
        parts += [f"Area {area_id}, expected product {expected_id}: {name}", image]
    parts.append(f"Answer for the {len(areas)} areas above, each inside its own magenta outline.")
    return parts


def frame_schema(area_ids: list[str], candidate_ids: list[str]) -> dict:
    """JSON schema of the answer: one entry per area (each entry is one provider entry of vlm.json plus area_id)."""
    properties = {
        "area_id": {"type": "string", "enum": list(area_ids)},
        "reason": {"type": "string"},
        "observed": {"type": "string", "enum": sorted(candidate_ids) + list(NO_PRODUCT)},
        "matches_expected": {"type": "boolean"},
        "fill_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "stock": {"type": "string", "enum": list(STOCK_VALUES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    item = {"type": "object", "properties": properties, "required": list(properties), "propertyOrdering": list(properties)}
    areas = {"type": "array", "items": item, "minItems": len(area_ids), "maxItems": len(area_ids)}
    return {"type": "object", "properties": {"areas": areas}, "required": ["areas"]}


def clean_answer(answer: dict, candidate_ids: list[str]) -> dict:
    """Check and normalise one parsed area entry. Raises ValueError when it does not follow the schema."""
    if answer.get("stock") not in STOCK_VALUES:
        raise ValueError(f"bad stock: {answer.get('stock')!r}")
    if answer.get("observed") not in (*candidate_ids, *NO_PRODUCT):
        raise ValueError(f"bad observed: {answer.get('observed')!r}")
    if not isinstance(answer.get("matches_expected"), bool):
        raise ValueError(f"bad matches_expected: {answer.get('matches_expected')!r}")
    return {
        "stock": answer["stock"],
        "fill_pct": min(100, max(0, int(answer["fill_pct"]))),
        "matches_expected": answer["matches_expected"],
        "observed": answer["observed"],
        "confidence": min(1.0, max(0.0, float(answer["confidence"]))),
        "reason": str(answer.get("reason", "")),
    }


def clean_frame_answer(answer: dict, area_ids: list[str], candidate_ids: list[str]) -> dict[str, dict | None]:
    """Split a parsed frame answer per area. A missing, duplicate or invalid entry gives None for that area only."""
    out: dict[str, dict | None] = dict.fromkeys(area_ids)
    for entry in answer.get("areas", []):
        area_id = entry.get("area_id") if isinstance(entry, dict) else None
        if area_id in out and out[area_id] is None:
            try:
                out[area_id] = clean_answer(entry, candidate_ids)
            except (ValueError, TypeError, KeyError):
                pass
    return out
