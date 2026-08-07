"""Default alignment scoring parameters, keyed by substrate and substrate type.

The two alignment stages in ``alignment.py`` use opposite scoring conventions,
and their parameters are NOT interchangeable:

    * Wagner-Fisher (``wf_*``) parameters are edit COSTS and are minimized, so
      positive values penalize. The match cost (0) and mismatch cost (+1) are
      hardcoded in ``wagner_fisher_affine`` and are not configurable; only the
      gap costs are exposed.
    * Smith-Waterman (``sw_*``) parameters are SCORES and are maximized, being
      summed into a running total. Rewards are positive and penalties are
      NEGATIVE -- including ``sw_mismatch``, which is added to the score on a
      mismatch rather than subtracted from it.

A profile is selected by the substrate (``--model``) plus whether the sample is
in vitro transcribed (``--ivt_alignment``). Profile names follow the checkpoint
naming convention used in ``configs/`` (e.g. ``BIOecoli``, ``IVTyeast``) so a
profile can be eyeballed against the model that produced the inference archive.
"""

import warnings
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class AlignParams:
    """Scoring parameters for one alignment profile.

    Attributes:
        wf_gap_open: Wagner-Fisher gap open cost (positive penalizes).
        wf_gap_extend: Wagner-Fisher gap extend cost (positive penalizes).
        sw_gap_open: Smith-Waterman gap open score (negative penalizes).
        sw_gap_extend: Smith-Waterman gap extend score (negative penalizes).
        sw_match: Smith-Waterman match score (positive rewards).
        sw_mismatch: Smith-Waterman mismatch score (negative penalizes).
        ident_threshold: Minimum identity for an alignment to be kept.
        max_query_length: Longest query, in bases, either stage will attempt.
            Applies to both Wagner-Fisher and Smith-Waterman. Both allocate
            three matrices of (reference length + 1) x (query length + 1), so
            cost grows with the query and a single pathological read can stall
            a worker for minutes and allocate hundreds of MB. A mature tRNA is
            under ~100 bases, so a query far above that is a bad predicted
            region rather than a real one; such reads are left unmapped.
    """

    wf_gap_open: float = 1.0
    wf_gap_extend: float = 0.7
    sw_gap_open: float = -6.0
    sw_gap_extend: float = -3.0
    sw_match: float = 3.0
    sw_mismatch: float = -3.0
    ident_threshold: float = 0.75
    max_query_length: int = 10000


# Substrate types, in the order they appear in a profile name.
BIOLOGICAL = "BIO"
IVT = "IVT"

# TODO: the BIO and IVT profiles are currently identical. IVT tRNA is unmodified
# and basecalls cleanly, whereas biological tRNA carries modifications that
# surface as systematic mismatches, so these are expected to diverge once the
# per-substrate-type optima have been measured.
ALIGN_PROFILES: Dict[Tuple[str, str], AlignParams] = {
    ("e_coli", BIOLOGICAL): AlignParams(),
    ("e_coli", IVT): AlignParams(),
    ("yeast", BIOLOGICAL): AlignParams(),
    ("yeast", IVT): AlignParams(),
}

# Maps the substrate as spelled by --model onto the fragment used in profile
# names and checkpoint filenames.
_SUBSTRATE_LABELS = {"e_coli": "ecoli", "yeast": "yeast"}


def profile_name(model: str, ivt_alignment: bool = False) -> str:
    """Build the display name of the profile selected by a model and IVT flag.

    params:
        model: Substrate as passed to --model ('yeast' or 'e_coli')
        ivt_alignment: True to name the in vitro transcribed profile

    returns: Profile name, e.g. 'BIOecoli' or 'IVTyeast'
    """
    substrate_type = IVT if ivt_alignment else BIOLOGICAL
    return f"{substrate_type}{_SUBSTRATE_LABELS.get(model, model)}"


def default_align_params(
    model: str,
    ivt_alignment: bool = False,
    *,
    wf_gap_open: Optional[float] = None,
    wf_gap_extend: Optional[float] = None,
    sw_gap_open: Optional[float] = None,
    sw_gap_extend: Optional[float] = None,
    sw_match: Optional[float] = None,
    sw_mismatch: Optional[float] = None,
    ident_threshold: Optional[float] = None,
    max_query_length: Optional[int] = None,
) -> AlignParams:
    """Look up a profile's parameters, applying any explicit overrides.

    params:
        model: Substrate as passed to --model ('yeast' or 'e_coli')
        ivt_alignment: True to select the in vitro transcribed profile
        wf_gap_open, wf_gap_extend, sw_gap_open, sw_gap_extend, sw_match,
        sw_mismatch, ident_threshold, max_query_length: An explicit value
        overrides that one field of the profile. None means 'use the profile
        value'.

    returns: An AlignParams with the profile values, overridden field by field

    raises:
        ValueError: If the model is not a known substrate
    """
    substrate_type = IVT if ivt_alignment else BIOLOGICAL
    try:
        params = ALIGN_PROFILES[(model, substrate_type)]
    except KeyError:
        substrates = sorted({m for m, _ in ALIGN_PROFILES})
        raise ValueError(
            f"{model} is not a recognized substrate, please choose from"
            f" {', '.join(substrates)}."
        ) from None

    overrides = {
        "wf_gap_open": wf_gap_open,
        "wf_gap_extend": wf_gap_extend,
        "sw_gap_open": sw_gap_open,
        "sw_gap_extend": sw_gap_extend,
        "sw_match": sw_match,
        "sw_mismatch": sw_mismatch,
        "ident_threshold": ident_threshold,
        "max_query_length": max_query_length,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(params, **overrides) if overrides else params


def validate_align_params(params: AlignParams) -> None:
    """Warn about parameters whose sign contradicts its scoring convention.

    Both scoring stages accumulate their parameters directly, so a sign error
    silently inverts the meaning of a penalty rather than raising. This catches
    the common cases, in particular a positive sw_mismatch, which rewards
    mismatches instead of penalizing them.

    params:
        params: The resolved parameters to check

    returns: None. Emits a UserWarning for each suspect value.
    """
    suspect = []

    if params.sw_mismatch > 0:
        suspect.append(
            f"sw_mismatch={params.sw_mismatch} is positive, which rewards"
            " mismatches; Smith-Waterman penalties must be negative"
        )
    if params.sw_match < 0:
        suspect.append(
            f"sw_match={params.sw_match} is negative, which penalizes matches"
        )
    if params.sw_gap_open > 0:
        suspect.append(
            f"sw_gap_open={params.sw_gap_open} is positive, which rewards gaps"
        )
    if params.sw_gap_extend > 0:
        suspect.append(
            f"sw_gap_extend={params.sw_gap_extend} is positive, which rewards gaps"
        )
    if params.wf_gap_open < 0:
        suspect.append(
            f"wf_gap_open={params.wf_gap_open} is negative, which rewards gaps;"
            " Wagner-Fisher costs must be positive"
        )
    if params.wf_gap_extend < 0:
        suspect.append(
            f"wf_gap_extend={params.wf_gap_extend} is negative, which rewards"
            " gaps; Wagner-Fisher costs must be positive"
        )
    if not 0.0 <= params.ident_threshold <= 1.0:
        suspect.append(
            f"ident_threshold={params.ident_threshold} is outside [0, 1];"
            " identity is a proportion"
        )
    if params.max_query_length < 100:
        suspect.append(
            f"max_query_length={params.max_query_length} is below the length of"
            " a mature tRNA, so every read will be left unmapped"
        )

    for message in suspect:
        warnings.warn(message, stacklevel=2)


def format_align_params(params: AlignParams) -> str:
    """Render parameters as a single line for the run log.

    Defaults are implicit once they come from a profile, so the run log is the
    only record of the values an alignment actually used.

    params:
        params: The resolved parameters to render

    returns: Space separated 'name=value' pairs
    """
    return " ".join(
        f"{f.name}={getattr(params, f.name)}" for f in fields(params)
    )
