def build_report(
    verdict: str, noise_detail: str, vit_detail: str, model_calibrated: bool, decision_band: str
) -> str:
    calibration_note = (
        ""
        if model_calibrated
        else " Neural model is in neutral mode because no forensic checkpoint is loaded."
    )
    confidence_note = (
        " Decision confidence is low; treat this as a screening result."
        if decision_band == "low"
        else ""
    )
    return (
        f"Verdict: {verdict}. "
        f"Statistical signal: {noise_detail} "
        f"Neural signal: {vit_detail}"
        f"{calibration_note}{confidence_note} "
        "This result combines sensor-noise residual analysis with transformer-based image forensics."
    )
