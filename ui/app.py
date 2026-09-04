"""
KYC Compliance Assistant - Gradio UI

Run from the project root:
    python ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.workflow import kyc_workflow

APPLICANTS_FILE = PROJECT_ROOT / "synthetic_documents" / "applicants.json"
DOCUMENT_ROOT = PROJECT_ROOT / "synthetic_documents"
PHOTO_ROOT = PROJECT_ROOT / "synthetic_photos"


# ---------------------------------------------------------------------------
# Applicant Data
# ---------------------------------------------------------------------------

def load_applicants() -> list[dict]:
    if not APPLICANTS_FILE.exists():
        raise FileNotFoundError(f"Applicant file not found: {APPLICANTS_FILE}")

    with APPLICANTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


APPLICANTS = load_applicants()
APPLICANT_MAP = {a["applicant_id"]: a for a in APPLICANTS}
APPLICANT_IDS = list(APPLICANT_MAP.keys())


# ---------------------------------------------------------------------------
# Document Helpers
# ---------------------------------------------------------------------------

def get_applicant_paths(applicant_id: str) -> dict[str, Path]:
    return {
        "aadhar": DOCUMENT_ROOT / applicant_id / "aadhar.png",
        "pan": DOCUMENT_ROOT / applicant_id / "pan.png",
        "address_proof": DOCUMENT_ROOT / applicant_id / "address_proof.png",
        "id_photo": PHOTO_ROOT / applicant_id / "id_photo.png",
        "selfie": PHOTO_ROOT / applicant_id / "selfie.png",
    }


def existing_file(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def load_document_previews(applicant_id: str):
    if not applicant_id:
        return None, None, None, None, None, "Select an applicant."

    paths = get_applicant_paths(applicant_id)
    scenario = APPLICANT_MAP.get(applicant_id, {}).get("scenario", "unknown")

    labels = {
        "aadhar": "Aadhaar",
        "pan": "PAN",
        "address_proof": "Address Proof",
        "id_photo": "ID Photo",
        "selfie": "Selfie",
    }

    document_status = [
        f"{'✓' if paths[key].is_file() else '✗'} {labels[key]}"
        + ("" if paths[key].is_file() else " — missing")
        for key in labels
    ]

    info = (
        f"### Applicant `{applicant_id}`\n"
        f"**Scenario:** `{scenario}`\n\n"
        + "\n".join(document_status)
    )

    return (
        existing_file(paths["aadhar"]),
        existing_file(paths["pan"]),
        existing_file(paths["address_proof"]),
        existing_file(paths["id_photo"]),
        existing_file(paths["selfie"]),
        info,
    )


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def safe(value: Any, default: str = "N/A") -> str:
    return default if value is None or value == "" else str(value)


def status_icon(status: Any) -> str:
    value = str(status or "").upper()

    if value in {
        "PASS",
        "MATCH",
        "CLEAR",
        "COMPLETE",
        "APPROVE",
        "HIGH",
        "EVIDENCE_FOUND",
    }:
        return "✓"

    if value in {
        "REVIEW",
        "LOW",
        "POTENTIAL_MATCH",
        "ANALYSIS_UNCERTAIN",
    }:
        return "⚠"

    if value in {
        "MISMATCH",
        "REJECT",
        "INCOMPLETE",
        "ERROR",
    }:
        return "✗"

    return "•"


def format_list(value: Any) -> str:
    if value in (None, "", [], {}):
        return "_None reported._"

    if isinstance(value, dict):
        lines = []

        for key, item in value.items():
            if item not in (None, "", [], {}):
                label = key.replace("_", " ").title()

                if isinstance(item, (dict, list)):
                    lines.append(
                        f"**{label}:**\n```json\n"
                        f"{json.dumps(item, indent=2, ensure_ascii=False, default=str)}\n```"
                    )
                else:
                    lines.append(f"- **{label}:** {item}")

        return "\n".join(lines) or "_None reported._"

    if isinstance(value, list):
        lines = []

        for item in value:
            if isinstance(item, dict):
                parts = []

                for key, item_value in item.items():
                    if item_value not in (None, "", [], {}):
                        parts.append(
                            f"**{key.replace('_', ' ').title()}:** {item_value}"
                        )

                lines.append("- " + " | ".join(parts))
            else:
                lines.append(f"- {item}")

        return "\n".join(lines) or "_None reported._"

    return safe(value)


def format_sources(sources: Any) -> str:
    if not sources:
        return "_No RBI policy citations available._"

    lines = []

    for source in sources:
        if not isinstance(source, dict):
            lines.append(f"- {source}")
            continue

        lines.append(
            f"- **Section:** {safe(source.get('section'))}  \n"
            f"  **Title:** {safe(source.get('title'))}  \n"
            f"  **PDF Page(s):** {safe(source.get('pages'))}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render Functions
# ---------------------------------------------------------------------------

def render_decision(decision_result: dict) -> str:
    if not decision_result:
        return "_Analysis has not been run yet._"

    decision = safe(
        decision_result.get("decision"),
        "UNKNOWN",
    ).upper()

    confidence = safe(
        decision_result.get("confidence"),
        "UNKNOWN",
    ).upper()

    icons = {
        "APPROVE": "🟢",
        "REVIEW": "🟠",
        "REJECT": "🔴",
        "MORE_DOCUMENTS": "🟡",
    }

    return (
        f"## {icons.get(decision, '⚪')} {decision}\n\n"
        f"**Confidence:** `{confidence}`\n\n"
        f"### Why?\n\n"
        f"{format_list(decision_result.get('reasons', []))}"
    )


def render_verification_status(final_state: dict) -> str:
    document_result = final_state.get("document_result", {})
    identity_result = final_state.get("identity_result", {})
    sanctions_result = final_state.get("sanctions_result", {})
    policy_result = final_state.get("policy_result", {})
    decision_result = final_state.get("decision_result", {})

    rows = [
        (
            "Document Verification",
            document_result.get("status", "NOT_RUN"),
        ),
        (
            "Identity Verification",
            identity_result.get("status", "NOT_RUN"),
        ),
        (
            "Photo Verification",
            identity_result.get("face_verification", {}).get(
                "assessment",
                "NOT_RUN",
            ),
        ),
        (
            "Sanctions Screening",
            sanctions_result.get("screening_status", "NOT_RUN"),
        ),
        (
            "Policy Compliance",
            policy_result.get("status", "NOT_RUN"),
        ),
        (
            "Decision Agent",
            decision_result.get("decision", "NOT_RUN"),
        ),
    ]

    lines = [
        "| Component | Status |",
        "|---|---|",
    ]

    for name, status in rows:
        lines.append(
            f"| **{name}** | "
            f"{status_icon(status)} `{safe(status, 'NOT RUN')}` |"
        )

    return "\n".join(lines)


def render_evidence(final_state: dict) -> str:
    document_result = final_state.get("document_result", {})
    identity_result = final_state.get("identity_result", {})
    sanctions_result = final_state.get("sanctions_result", {})
    policy_result = final_state.get("policy_result", {})
    decision_result = final_state.get("decision_result", {})

    document_assessment = decision_result.get(
        "document_assessment",
        {},
    )

    identity_assessment = decision_result.get(
        "identity_assessment",
        {},
    )

    sanctions_assessment = decision_result.get(
        "sanctions_assessment",
        {},
    )

    policy_assessment = decision_result.get(
        "policy_assessment",
        {},
    )

    identity_verification = identity_result.get(
        "identity_verification",
        {},
    )

    face_verification = identity_result.get(
        "face_verification",
        {},
    )

    return f"""
### 📄 Document Verification

**Status:** `{document_assessment.get('status', document_result.get('status', 'N/A'))}`

**Documents found**

{format_list(
    document_assessment.get(
        'documents_found',
        document_result.get('documents_found', []),
    )
)}

**Missing documents**

{format_list(
    decision_result.get(
        'missing_documents',
        document_result.get('missing_documents', []),
    )
)}

**Visual / tampering evidence**

{format_list(
    document_assessment.get(
        'visual_analysis',
        document_result.get('visual_analysis', []),
    )
)}

---

### 🪪 Identity Verification

**Status:** `{identity_assessment.get('status', identity_result.get('status', 'N/A'))}`

| Check | Result |
|---|---|
| Name similarity | `{safe(identity_assessment.get('name_similarity', identity_verification.get('name_similarity')))}`
| DOB match | `{safe(identity_assessment.get('dob_match', identity_verification.get('dob_match')))}`
| Address similarity | `{safe(identity_assessment.get('address_similarity', identity_verification.get('address_similarity')))}`
    
**Photo Verification**

**Assessment:** `{safe(face_verification.get('assessment'))}`

**Face similarity:** `{safe(face_verification.get('face_similarity'))}`

**Observations**

{format_list(face_verification.get('observations', []))}

---

### 🛡️ Sanctions Screening

**Status:** `{sanctions_assessment.get('status', sanctions_result.get('screening_status', 'N/A'))}`

**Candidates**

{format_list(
    sanctions_assessment.get(
        'candidates',
        sanctions_result.get('candidates', []),
    )
)}

**Evidence**

{format_list(sanctions_assessment.get('evidence', []))}

---

### 📚 RBI Policy Compliance

**Status:** `{policy_assessment.get('status', policy_result.get('status', 'N/A'))}`

**Policy basis**

{safe(
    decision_result.get(
        'policy_basis',
        policy_result.get('answer', ''),
    ),
    '_No policy answer available._',
)}

**RBI citations**

{format_sources(
    policy_assessment.get(
        'citations',
        policy_result.get('sources', []),
    )
)}
"""


def render_audit(final_state: dict) -> str:
    return json.dumps(
        final_state.get("decision_result", {}),
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def render_workflow_log(final_state: dict) -> str:
    messages = final_state.get("messages", [])

    lines = [
        f"- {message}"
        for message in messages
        if message
    ]

    if final_state.get("sanctions_review_attempted", False):
        lines.append(
            "- 🔄 One-time sanctions feedback review was executed."
        )

    return (
        "\n".join(lines)
        if lines
        else "_No workflow messages._"
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_application(applicant_id: str):
    if not applicant_id:
        message = (
            "## ⚠️ Select an applicant\n\n"
            "Please select an applicant."
        )

        return (
            message,
            message,
            message,
            "{}",
            "_No workflow execution._",
            "",
        )

    applicant = APPLICANT_MAP.get(applicant_id)

    if applicant is None:
        message = (
            "## ❌ Applicant Not Found\n\n"
            f"`{applicant_id}` does not exist."
        )

        return (
            message,
            message,
            message,
            "{}",
            message,
            "",
        )

    paths = get_applicant_paths(applicant_id)

    initial_state = {
        "applicant_id": applicant_id,
        "expected_profile": applicant,

        "document_paths": {
            "aadhar": str(paths["aadhar"]),
            "pan": str(paths["pan"]),
            "address_proof": str(paths["address_proof"]),
        },

        "photo_paths": {
            "id_photo": str(paths["id_photo"]),
            "selfie": str(paths["selfie"]),
        },

        "document_result": {},
        "identity_result": {},
        "sanctions_result": {},
        "policy_result": {},
        "decision_result": {},

        "next_action": None,
        "messages": [],
        "error": None,
        "sanctions_review_attempted": False,
    }

    try:
        final_state = kyc_workflow.invoke(initial_state)

    except Exception as exc:
        error = (
            "## ❌ Workflow Execution Failed\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`"
        )

        return (
            error,
            error,
            error,
            "{}",
            error,
            "",
        )

    decision_result = final_state.get(
        "decision_result",
        {},
    )

    decision = decision_result.get(
        "decision",
        "UNKNOWN",
    )

    return (
        render_decision(decision_result),
        render_verification_status(final_state),
        render_evidence(final_state),
        render_audit(final_state),
        render_workflow_log(final_state),
        f"### ✅ Analysis Complete\n\n"
        f"Applicant `{applicant_id}` → **{decision}**",
    )


# ---------------------------------------------------------------------------
# NEW:
# Reset all analysis outputs when applicant changes
# ---------------------------------------------------------------------------

def handle_applicant_change(applicant_id: str):
    """
    Load the newly selected applicant's documents AND reset all
    previous KYC analysis outputs.

    This prevents the previous applicant's KYC decision,
    verification status, evidence, audit trail, and workflow
    information from remaining on screen.
    """

    # Load new applicant documents
    (
        aadhar,
        pan,
        address_proof,
        id_photo,
        selfie,
        applicant_info,
    ) = load_document_previews(applicant_id)

    # Reset analysis section
    decision_default = (
        "_Analysis has not been run yet._"
    )

    status_default = """| Component | Status |
|---|---|
| **Document Verification** | • `NOT RUN` |
| **Identity Verification** | • `NOT RUN` |
| **Photo Verification** | • `NOT RUN` |
| **Sanctions Screening** | • `NOT RUN` |
| **Policy Compliance** | • `NOT RUN` |
| **Decision Agent** | • `NOT RUN` |"""

    evidence_default = (
        "_Run an analysis to see structured evidence._"
    )

    audit_default = "{}"

    workflow_default = (
        "_No workflow execution yet._"
    )

    summary_default = ""

    return (
        # Document previews
        aadhar,
        pan,
        address_proof,
        id_photo,
        selfie,
        applicant_info,

        # Reset previous applicant analysis
        decision_default,
        status_default,
        evidence_default,
        audit_default,
        workflow_default,
        summary_default,
    )


# ---------------------------------------------------------------------------
# UI Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#app-header {
    text-align: center;
    margin-bottom: 15px;
}

#app-header h1 {
    font-size: 32px;
    margin-bottom: 5px;
}

#app-header p {
    font-size: 16px;
    opacity: 0.8;
}

.document-preview img {
    object-fit: contain;
}
"""


# ---------------------------------------------------------------------------
# Gradio Application
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="KYC Compliance Assistant",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
# 🏦 KYC Compliance Assistant

### AI-Powered Customer Onboarding & KYC Verification

**LangGraph** orchestration • Document OCR • Identity Verification •
Face Verification • OFAC Screening • RBI Policy RAG
""",
        elem_id="app-header",
    )

    # -----------------------------------------------------------------------
    # Applicant Selection
    # -----------------------------------------------------------------------

    with gr.Row():

        with gr.Column(scale=3):
            applicant_dropdown = gr.Dropdown(
                choices=APPLICANT_IDS,
                value=APPLICANT_IDS[0] if APPLICANT_IDS else None,
                label="Applicant ID",
                info="Select a synthetic applicant.",
            )

        with gr.Column(scale=1):
            analyze_button = gr.Button(
                "🔍 Analyze Application",
                variant="primary",
                size="lg",
            )

    applicant_info = gr.Markdown(
        "Select an applicant to preview the submitted evidence."
    )

    # -----------------------------------------------------------------------
    # KYC Evidence
    # -----------------------------------------------------------------------

    gr.Markdown("## 📁 KYC Evidence")

    with gr.Row():

        aadhar_image = gr.Image(
            label="Aadhaar",
            type="filepath",
            height=220,
            interactive=False,
            elem_classes=["document-preview"],
        )

        pan_image = gr.Image(
            label="PAN",
            type="filepath",
            height=220,
            interactive=False,
            elem_classes=["document-preview"],
        )

        address_image = gr.Image(
            label="Address Proof",
            type="filepath",
            height=220,
            interactive=False,
            elem_classes=["document-preview"],
        )

    with gr.Row():

        id_photo_image = gr.Image(
            label="ID Photo",
            type="filepath",
            height=220,
            interactive=False,
            elem_classes=["document-preview"],
        )

        selfie_image = gr.Image(
            label="Selfie",
            type="filepath",
            height=220,
            interactive=False,
            elem_classes=["document-preview"],
        )

    # -----------------------------------------------------------------------
    # KYC Decision
    # -----------------------------------------------------------------------

    gr.Markdown("## 🎯 KYC Decision")

    decision_output = gr.Markdown(
        "_Analysis has not been run yet._"
    )

    # -----------------------------------------------------------------------
    # Verification Status
    # -----------------------------------------------------------------------

    gr.Markdown("## 🔎 Verification Status")

    status_output = gr.Markdown(
        """| Component | Status |
|---|---|
| **Document Verification** | • `NOT RUN` |
| **Identity Verification** | • `NOT RUN` |
| **Photo Verification** | • `NOT RUN` |
| **Sanctions Screening** | • `NOT RUN` |
| **Policy Compliance** | • `NOT RUN` |
| **Decision Agent** | • `NOT RUN` |"""
    )

    # -----------------------------------------------------------------------
    # Evidence & Reasoning
    # -----------------------------------------------------------------------

    gr.Markdown("## 🧾 Evidence & Reasoning")

    evidence_output = gr.Markdown(
        "_Run an analysis to see structured evidence._"
    )

    # -----------------------------------------------------------------------
    # Audit Trail
    # -----------------------------------------------------------------------

    with gr.Accordion(
        "📋 Full Audit Trail",
        open=False,
    ):
        audit_output = gr.Code(
            value="{}",
            language="json",
            label="Decision Audit JSON",
        )

    # -----------------------------------------------------------------------
    # Workflow
    # -----------------------------------------------------------------------

    with gr.Accordion(
        "⚙️ Workflow Execution",
        open=False,
    ):
        workflow_output = gr.Markdown(
            "_No workflow execution yet._"
        )

    analysis_summary = gr.Markdown("")

    # -----------------------------------------------------------------------
    # IMPORTANT FIX:
    # Applicant dropdown now resets previous analysis outputs
    # -----------------------------------------------------------------------

    applicant_dropdown.change(
        fn=handle_applicant_change,
        inputs=[applicant_dropdown],
        outputs=[
            # New applicant documents
            aadhar_image,
            pan_image,
            address_image,
            id_photo_image,
            selfie_image,
            applicant_info,

            # Reset previous applicant's analysis
            decision_output,
            status_output,
            evidence_output,
            audit_output,
            workflow_output,
            analysis_summary,
        ],
    )

    # -----------------------------------------------------------------------
    # Analyze Button
    # -----------------------------------------------------------------------

    analyze_button.click(
        fn=analyze_application,
        inputs=[applicant_dropdown],
        outputs=[
            decision_output,
            status_output,
            evidence_output,
            audit_output,
            workflow_output,
            analysis_summary,
        ],
    )

    # -----------------------------------------------------------------------
    # Initial Page Load
    # -----------------------------------------------------------------------

    demo.load(
        fn=load_document_previews,
        inputs=[applicant_dropdown],
        outputs=[
            aadhar_image,
            pan_image,
            address_image,
            id_photo_image,
            selfie_image,
            applicant_info,
        ],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        share=True,
    )
