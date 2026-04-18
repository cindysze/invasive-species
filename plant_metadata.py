"""Taxon metadata: invasive risk scoped to San Diego County and short blurbs for the demo."""

from __future__ import annotations

from config import REGION_LONG, REGION_SHORT

# Species on Cal-IPC / regional lists that are also common problems in coastal Southern California,
# including San Diego County open space, canyons, and wildland–urban edges.
# Used only to enrich predictions — always cite authoritative sources for management decisions.
SD_HIGH_IMPACT = frozenset(
    {
        "Arundo donax",
        "Asparagus asparagoides",
        "Brassica tournefortii",
        "Carpobrotus edulis",
        "Centaurea melitensis",
        "Centaurea solstitialis",
        "Cenchrus setaceus",
        "Cortaderia selloana",
        "Cortaderia jubata",
        "Cynara cardunculus",
        "Dittrichia graveolens",
        "Ehrharta erecta",
        "Eichhornia crassipes",
        "Genista monspessulana",
        "Glebionis coronaria",
        "Helminthotheca echioides",
        "Hirschfeldia incana",
        "Mesembryanthemum crystallinum",
        "Nicotiana glauca",
        "Oxalis pes-caprae",
        "Ricinus communis",
        "Schinus molle",
        "Schinus terebinthifolia",
        "Tamarix",
        "Volutaria tubuliflora",
        "Asphodelus fistulosus",
        "Foeniculum vulgare",
    }
)

SD_MODERATE_OR_WEEDY = frozenset(
    {
        "Tropaeolum majus",
        "Medicago polymorpha",
        "Sonchus oleraceus",
        "Erigeron canadensis",
        "Hypochaeris glabra",
        "Marrubium vulgare",
        "Silybum marianum",
        "Cirsium vulgare",
        "Erodium cicutarium",
    }
)


def invasive_label(scientific_name: str) -> tuple[str, str]:
    """
    Returns (short_label, explanation) for display.
    """
    name = (scientific_name or "").strip()
    if not name:
        return "Unknown", "No scientific name available for this record."
    if name in SD_HIGH_IMPACT:
        return (
            f"Listed / widely treated as invasive in {REGION_SHORT}",
            "Often appears on Cal-IPC, county weed lists, or land-manager priority species lists for "
            f"{REGION_SHORT}. Confirm ID before control; follow local regulations and best practices for your site.",
        )
    if name in SD_MODERATE_OR_WEEDY:
        return (
            f"Weedy or patchily invasive in {REGION_SHORT}",
            "Can be aggressive in disturbed soils, roadsides, or gardens. "
            "Verify with San Diego native-plant references or UC Cooperative Extension if managing on public land.",
        )
    return (
        f"Not auto-flagged as a top-tier {REGION_SHORT} invader here",
        "This species may still be problematic in specific canyons, wetlands, or coastal sites, "
        "or the ID may be wrong. Cross-check with local inventories and expert ID if it matters for management.",
    )


def build_record(taxon_id: str, scientific_name: str, common_name: str) -> dict:
    short, expl = invasive_label(scientific_name)
    return {
        "taxon_id": taxon_id,
        "scientific_name": scientific_name,
        "common_name": common_name or "",
        "study_region": REGION_LONG,
        "invasive_summary": short,
        "invasive_detail": expl,
    }
