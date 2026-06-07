import re
import html
import os
import markdown
from fpdf import FPDF

class PremiumPDF(FPDF):
    def header(self):
        # Only add header from page 2 onwards
        if self.page_no() > 1:
            self.set_font("Arial", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "Customer Insights & Recommendation System | Interview Guide", align="R")
            self.ln(8)
            # Draw a subtle separator line under header
            self.set_draw_color(220, 220, 220)
            self.line(self.l_margin, self.t_margin + 5, 210 - self.r_margin, self.t_margin + 5)

    def footer(self):
        # Footer on every page
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def strip_emojis_and_unsupported(text):
    # Emojis and other non-standard symbols are stripped to prevent FPDF font failures
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002700-\U000027bf"  # dingbats
        "\U00002600-\U000026ff"  # miscellaneous symbols
        "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
        "\U0001fa70-\U0001faff"  # symbols and pictographs extended
        "\u2600-\u27bf"
        "\U0001f400-\U0001f5ff"
        "\U0001f600-\U0001f6ff"
        "\U0001f900-\U0001f9ff"
        "\u2705\u274c\u26a0\u2139\u2714\u274e\u2705"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    # Clean up bullet list characters that might be weird
    text = text.replace("🧠", "").replace("🔄", "").replace("🚨", "").replace("❓", "").replace("💡", "")
    text = text.replace("➕", "").replace("➖", "").replace("❌", "").replace("👉", "").replace("🔙", "")
    text = text.replace("📊", "").replace("✅", "").replace("🌟", "")
    return text

def clean_special_chars(text):
    # Replace curly quotes and dashes with ASCII equivalents to prevent FPDF built-in font errors
    replacements = {
        "–": "-", # En-dash
        "—": "--", # Em-dash
        "“": '"', # Left double quote
        "”": '"', # Right double quote
        "‘": "'", # Left single quote
        "’": "'", # Right single quote
        "…": "...", # Ellipsis
        "−": "-", # Mathematical minus
        "²": "^2", # Superscript 2
        "₂": "2", # Subscript 2
        "\xa0": " ", # Non-breaking space
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def parse_code_blocks(text):
    # Convert code blocks to styled HTML block (no tables to avoid FPDF parser issues with nested tags)
    def replacer(match):
        code = match.group(2).strip()
        escaped_code = html.escape(code)
        # Indent each line by 4 spaces and convert to non-breaking spaces
        indented_lines = []
        for line in escaped_code.split("\n"):
            indented_lines.append("&nbsp;&nbsp;&nbsp;&nbsp;" + line.replace(" ", "&nbsp;"))
        html_code = "<br>".join(indented_lines)
        return (
            f'<br><hr><font face="Courier" size="2" color="#2d3748"><b>Code:</b><br>{html_code}</font><hr><br>'
        )
    pattern = r"```(python|bash|diff|mermaid|markdown)?\n(.*?)\n```"
    return re.sub(pattern, replacer, text, flags=re.DOTALL)

def parse_callouts(text):
    # Match markdown blockquotes and format warning/note alerts as simple text blocks (no tables)
    lines = text.split("\n")
    processed_lines = []
    in_quote = False
    quote_lines = []

    for line in lines:
        if line.startswith(">"):
            in_quote = True
            quote_lines.append(line[1:].strip())
        else:
            if in_quote:
                quote_text = " ".join(quote_lines)
                label = "INFO: "
                color = "#2b6cb0" # Blue

                if "[!WARNING]" in quote_text or "[!CAUTION]" in quote_text:
                    label = "WARNING: "
                    color = "#e53e3e" # Red
                    quote_text = quote_text.replace("[!WARNING]", "").replace("[!CAUTION]", "").strip()
                elif "[!NOTE]" in quote_text or "[!TIP]" in quote_text or "[!IMPORTANT]" in quote_text:
                    label = "NOTE: "
                    color = "#3182ce" # Blue
                    quote_text = quote_text.replace("[!NOTE]", "").replace("[!TIP]", "").replace("[!IMPORTANT]", "").strip()

                quote_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", quote_text)
                
                callout_html = (
                    f'<br><font size="2" color="{color}"><b>{label}</b><i>{quote_text}</i></font><br>'
                )
                processed_lines.append(callout_html)
                quote_lines = []
                in_quote = False
            processed_lines.append(line)

    return "\n".join(processed_lines)

def build_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pre-process content
    content = strip_emojis_and_unsupported(content)
    content = clean_special_chars(content)
    content = parse_code_blocks(content)
    content = parse_callouts(content)

    # Convert remaining markdown to HTML
    html_content = markdown.markdown(content, extensions=['tables'])

    # Style standard headers with dark corporate palette
    html_content = html_content.replace("<h1>", '<h1 style="color: #1a365d; font-size: 20px; font-weight: bold; margin-top: 15px;">')
    html_content = html_content.replace("<h2>", '<h2 style="color: #2b6cb0; font-size: 16px; font-weight: bold; margin-top: 12px;">')
    html_content = html_content.replace("<h3>", '<h3 style="color: #4a5568; font-size: 12px; font-weight: bold; margin-top: 8px;">')
    html_content = html_content.replace("<h4>", '<h4 style="color: #4a5568; font-size: 10px; font-weight: bold; margin-top: 6px;">')

    pdf = PremiumPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)

    # Load system TrueType fonts for clean Unicode rendering (prevents crash on curly quotes)
    pdf.add_font("Arial", "", "/System/Library/Fonts/Supplemental/Arial.ttf")
    pdf.add_font("Arial", "B", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    pdf.add_font("Arial", "I", "/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    pdf.add_font("Arial", "BI", "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf")
    
    # 1. Generate Cover Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(26, 54, 93) # Navy
    pdf.ln(40)
    pdf.cell(0, 15, "AI-Driven Customer Insights", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "& Recommendation System", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(43, 108, 176) # Dark Blue
    pdf.cell(0, 10, "INTERVIEW PREPARATION GUIDE", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Draw horizontal separator line on cover page
    pdf.ln(10)
    pdf.set_draw_color(43, 108, 176)
    pdf.set_line_width(0.8)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    
    pdf.ln(30)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 8, "A complete codebase walkthrough, system data flow mapping,", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "critical logic analysis, and 200 curated technical & behavioral Q&As.", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(40)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, "Prepared for: Technical Interview Prep", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Target Role: Machine Learning Engineer / Full-Stack Developer", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Start guide content on page 2
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(45, 55, 72) # Charcoal grey
    
    # Write the compiled HTML content
    pdf.write_html(html_content)
    
    # Export the PDF
    pdf.output(pdf_path)
    print(f"Success: PDF generated successfully at {pdf_path}")

if __name__ == "__main__":
    md_file = "/Users/anshrathore/.gemini/antigravity-ide/brain/0489659a-0a3a-4543-aa19-40107be52fd7/interview_preparation_guide.md"
    pdf_file = "/Users/anshrathore/Desktop/AI-Driven-Customer-Insights-and-Recommendation-System/interview_preparation_guide.pdf"
    build_pdf(md_file, pdf_file)
