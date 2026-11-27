"""Phase 12 (Day111-Day120, Gate G12) sealed final evaluation, clean-checkout verifier, and release manifest builder."""

from scratchlm.release.final_release import (
    build_release_candidate_manifest,
    evaluate_sealed_final_test_once,
    generate_all_phase12_artifacts,
    verify_clean_checkout_rehearsal,
)

__all__ = [
    "build_release_candidate_manifest",
    "evaluate_sealed_final_test_once",
    "generate_all_phase12_artifacts",
    "verify_clean_checkout_rehearsal",
]
