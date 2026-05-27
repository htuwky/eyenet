from __future__ import annotations


def test_core_modules_import() -> None:
    import eyenet.data.subject_summary  # noqa: F401
    import eyenet.models.dual_stream  # noqa: F401
    import eyenet.training.summary_encoder_dual_stream  # noqa: F401
