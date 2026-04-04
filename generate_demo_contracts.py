"""Generates two demo contract PDFs for testing SemanticContractDiff."""

import fitz  # PyMuPDF

LINE_HEIGHT = 14
FONT_SIZE_HEADING = 11
FONT_SIZE_BODY = 10
MARGIN = 50
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN


SECTION_HEIGHT = 80  # estimated height per section in points


def make_pdf(filename: str, sections: list) -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN

    for title, body in sections:
        text = f"{title}\n{body}"
        # Estimate lines needed: ~90 chars per line at font size 10
        estimated_lines = max(4, len(text) // 90 + 2)
        block_height = estimated_lines * LINE_HEIGHT + 10

        if y + block_height > PAGE_HEIGHT - MARGIN:
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            y = MARGIN

        rect = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + block_height)
        overflow = page.insert_textbox(
            rect,
            text,
            fontsize=FONT_SIZE_BODY,
            fontname="helv",
            color=(0, 0, 0),
        )
        # If text overflowed, use a bigger rect
        if overflow < 0:
            block_height = block_height * 2
            rect = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + block_height)
            page.insert_textbox(
                rect,
                text,
                fontsize=FONT_SIZE_BODY,
                fontname="helv",
                color=(0, 0, 0),
            )

        y += block_height

        # Insert a blank line character so get_text() produces \n\n between blocks
        if y + LINE_HEIGHT < PAGE_HEIGHT - MARGIN:
            page.insert_text((MARGIN, y + LINE_HEIGHT), " ",
                             fontsize=FONT_SIZE_BODY, fontname="helv")
        y += LINE_HEIGHT * 2

    doc.save(filename)
    doc.close()
    print(f"Written: {filename}")


VERSION_A = [
    (
        "SOFTWARE SERVICES AGREEMENT - VERSION A",
        "This Software Services Agreement (\"Agreement\") is entered into as of January 1, 2024, "
        "by and between Acme Corp, a Delaware corporation (\"Provider\"), and Beta LLC, a California "
        "limited liability company (\"Customer\")."
    ),
    (
        "1. Services",
        "Provider agrees to deliver software development and maintenance services as described in each "
        "Statement of Work (\"SOW\") mutually executed by the parties. Provider shall assign a dedicated "
        "project manager to each SOW and will commence work within five (5) business days of execution. "
        "Provider will use commercially reasonable efforts to meet all deadlines set forth in each SOW."
    ),
    (
        "2. Payment Terms",
        "Customer shall pay all invoices within thirty (30) days of the invoice date. All fees are "
        "denominated and payable in United States Dollars (USD). Late payments shall accrue interest "
        "at a rate of one and a half percent (1.5%) per month on the outstanding balance. Provider "
        "reserves the right to suspend services after sixty (60) days of non-payment."
    ),
    (
        "3. Limitation of Liability",
        "In no event shall either party be liable to the other for any indirect, incidental, special, "
        "consequential, or punitive damages. Provider's total cumulative liability arising out of or "
        "related to this Agreement shall not exceed Ten Thousand United States Dollars (USD $10,000), "
        "regardless of the cause of action or the theory of liability."
    ),
    (
        "4. Term and Renewal",
        "This Agreement shall commence on the Effective Date and continue for a period of one (1) year "
        "(\"Initial Term\"). Upon expiration of the Initial Term, this Agreement shall automatically renew "
        "for successive one-year periods unless either party provides written notice of non-renewal at "
        "least thirty (30) days prior to the end of the then-current term."
    ),
    (
        "5. Termination for Convenience",
        "Either party may terminate this Agreement for any reason upon thirty (30) days prior written "
        "notice to the other party. Upon termination, Customer shall pay Provider for all services "
        "rendered up to and including the effective date of termination. Provider will deliver all "
        "work product completed as of the termination date."
    ),
    (
        "6. Intellectual Property",
        "All work product, inventions, and deliverables created by Provider under this Agreement "
        "shall be considered works made for hire and shall be the sole and exclusive property of Customer. "
        "Provider retains ownership of all pre-existing tools, frameworks, and methodologies used in "
        "delivering the services."
    ),
    (
        "7. Confidentiality",
        "Each party agrees to hold the other party's Confidential Information in strict confidence and "
        "not to disclose such information to any third party without prior written consent. This obligation "
        "shall survive the termination of this Agreement for a period of two (2) years."
    ),
    (
        "8. Governing Law and Dispute Resolution",
        "This Agreement shall be governed by and construed in accordance with the laws of the State of "
        "California, without regard to its conflict of law provisions. Any disputes arising under this "
        "Agreement shall be resolved through binding arbitration in San Francisco, California, under the "
        "rules of the American Arbitration Association."
    ),
    (
        "9. Indemnification",
        "Each party shall indemnify and hold harmless the other party from any third-party claims arising "
        "out of that party's gross negligence or willful misconduct. Provider shall have no indemnification "
        "obligation for claims arising from Customer's misuse of the deliverables."
    ),
    (
        "10. Entire Agreement",
        "This Agreement, together with all SOWs and exhibits, constitutes the entire agreement between "
        "the parties with respect to its subject matter and supersedes all prior agreements, representations, "
        "and understandings. This Agreement may not be amended except by a written instrument signed by "
        "authorized representatives of both parties."
    ),
]

VERSION_B = [
    (
        "SOFTWARE SERVICES AGREEMENT - VERSION B",
        "This Software Services Agreement (\"Agreement\") is entered into as of January 1, 2024, "
        "by and between Acme Corp, a Delaware corporation (\"Provider\"), and Beta LLC, a California "
        "limited liability company (\"Customer\")."
    ),
    (
        "1. Services",
        "Provider agrees to deliver software development and maintenance services as described in each "
        "Statement of Work (\"SOW\") mutually executed by the parties. Provider shall assign a dedicated "
        "project manager to each SOW and will commence work within five (5) business days of execution. "
        "Provider will use commercially reasonable efforts to meet all deadlines set forth in each SOW."
    ),
    (
        "2. Payment Terms",
        # Payment window halved, currency changed, interest rate more than doubled, suspension window halved
        "Customer shall pay all invoices within fifteen (15) days of the invoice date. All fees are "
        "denominated and payable in Euros (EUR) at the exchange rate on the invoice date. Late payments "
        "shall accrue interest at a rate of three and a half percent (3.5%) per month on the outstanding "
        "balance. Provider reserves the right to suspend services after thirty (30) days of non-payment."
    ),
    (
        "3. Limitation of Liability",
        # Cap increased 50x, punitive damages carve-out removed, gross negligence exception added
        "Provider's total cumulative liability arising out of or related to this Agreement shall not "
        "exceed Five Hundred Thousand United States Dollars (USD $500,000), regardless of the cause of "
        "action or the theory of liability. This limitation shall not apply to damages arising from "
        "Provider's gross negligence, fraud, or wilful misconduct."
    ),
    (
        "4. Term and Renewal",
        # Auto-renewal removed entirely, notice period tripled to 90 days
        "This Agreement shall commence on the Effective Date and continue for a period of one (1) year "
        "(\"Initial Term\"). This Agreement shall NOT automatically renew upon expiration. To extend the "
        "Agreement, the parties must execute a written renewal at least ninety (90) days prior to the "
        "end of the then-current term. Failure to execute a renewal will result in expiration of this Agreement."
    ),
    (
        "5. Termination for Convenience",
        # Notice tripled to 90 days, early termination fee of 3 months added
        "Either party may terminate this Agreement for any reason upon ninety (90) days prior written "
        "notice to the other party. If Customer terminates for convenience prior to the end of the Initial "
        "Term, Customer shall pay Provider an early termination fee equal to three (3) months of the "
        "average monthly fees paid in the six months preceding termination. Provider will deliver all "
        "work product completed as of the termination date."
    ),
    (
        "6. Intellectual Property",
        "All work product, inventions, and deliverables created by Provider under this Agreement "
        "shall be considered works made for hire and shall be the sole and exclusive property of Customer. "
        "Provider retains ownership of all pre-existing tools, frameworks, and methodologies used in "
        "delivering the services."
    ),
    (
        "7. Confidentiality",
        # Survival period extended from 2 to 3 years
        "Each party agrees to hold the other party's Confidential Information in strict confidence and "
        "not to disclose such information to any third party without prior written consent. This obligation "
        "shall survive the termination of this Agreement for a period of three (3) years."
    ),
    (
        "8. Governing Law and Dispute Resolution",
        # California -> New York, arbitration -> litigation, SF -> NYC
        "This Agreement shall be governed by and construed in accordance with the laws of the State of "
        "New York, without regard to its conflict of law provisions. Any disputes arising under this "
        "Agreement shall be resolved exclusively in the state or federal courts located in New York City, "
        "New York. Each party irrevocably consents to the personal jurisdiction of such courts."
    ),
    (
        "9. Indemnification",
        # Broad indemnification added including IP infringement and data protection
        "Provider shall indemnify, defend, and hold harmless Customer and its officers, directors, and "
        "employees from any third-party claims, damages, and expenses (including reasonable attorneys fees) "
        "arising out of: (a) Provider's gross negligence or wilful misconduct; (b) any claim that the "
        "deliverables infringe a third party's intellectual property rights; or (c) Provider's breach of "
        "its data protection obligations. Customer's indemnification obligations are limited to claims "
        "arising solely from Customer's misuse of the deliverables."
    ),
    (
        "10. Entire Agreement",
        "This Agreement, together with all SOWs and exhibits, constitutes the entire agreement between "
        "the parties with respect to its subject matter and supersedes all prior agreements, representations, "
        "and understandings. This Agreement may not be amended except by a written instrument signed by "
        "authorized representatives of both parties."
    ),
]

if __name__ == "__main__":
    make_pdf("demo_contract_v1.pdf", VERSION_A)
    make_pdf("demo_contract_v2.pdf", VERSION_B)
    print("\nDone. Major changes between versions:")
    print("  Clause 2 - Payment Terms: 30->15 days, USD->EUR, 1.5%->3.5% interest")
    print("  Clause 3 - Liability cap: $10,000 -> $500,000")
    print("  Clause 4 - Term: auto-renewal removed, notice 30->90 days")
    print("  Clause 5 - Termination: notice 30->90 days, early termination fee added")
    print("  Clause 8 - Governing law: California->New York, arbitration->litigation")
    print("  Clause 9 - Indemnification: significantly expanded to include IP and data protection")
