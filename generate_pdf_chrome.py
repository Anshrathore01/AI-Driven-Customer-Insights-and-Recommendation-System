import re
import os
import subprocess
import markdown

def parse_callouts(html_content):
    # Convert blockquote annotations into styled div callouts
    # We find <blockquote>...</blockquote> and examine if they have [!NOTE], [!WARNING], etc.
    def replacer(match):
        content = match.group(1).strip()
        callout_type = "note"
        border_color = "#3b82f6"
        bg_color = "#eff6ff"
        text_color = "#1e40af"
        title = "NOTE"

        if "[!WARNING]" in content or "[!CAUTION]" in content:
            callout_type = "warning"
            border_color = "#ef4444"
            bg_color = "#fef2f2"
            text_color = "#991b1b"
            title = "WARNING"
            content = content.replace("[!WARNING]", "").replace("[!CAUTION]", "").strip()
        elif "[!NOTE]" in content or "[!TIP]" in content or "[!IMPORTANT]" in content:
            callout_type = "note"
            border_color = "#3b82f6"
            bg_color = "#eff6ff"
            text_color = "#1e40af"
            title = "NOTE"
            content = content.replace("[!NOTE]", "").replace("[!TIP]", "").replace("[!IMPORTANT]", "").strip()

        return (
            f'<div class="callout callout-{callout_type}" style="border-left: 4px solid {border_color}; background-color: {bg_color}; color: {text_color}; padding: 12px 16px; margin: 1.5em 0; border-radius: 4px; page-break-inside: avoid;">'
            f'<div style="font-weight: bold; font-size: 0.85em; margin-bottom: 4px; text-transform: uppercase; tracking: 0.05em;">{title}</div>'
            f'<div>{content}</div>'
            f'</div>'
        )

    return re.sub(r"<blockquote>(.*?)</blockquote>", replacer, html_content, flags=re.DOTALL)

def build_beautiful_pdf(md_path, html_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    # We support tables, fenced code blocks, and toc
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])
    
    # Process Callouts
    html_body = parse_callouts(html_body)

    # Build the full HTML document with a beautiful CSS stylesheet
    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Driven Customer Insights & Recommendation System - Interview Guide</title>
    <!-- Load modern typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    
    <style>
        @page {{
            size: A4;
            margin: 20mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            color: #1e293b;
            line-height: 1.6;
            font-size: 10.5pt;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        
        /* Cover Page Styling */
        .cover-page {{
            page-break-after: always;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-sizing: border-box;
            padding-top: 5cm;
        }}
        
        .cover-title {{
            font-size: 32pt;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
            margin: 0 0 10px 0;
            letter-spacing: -0.02em;
        }}
        
        .cover-subtitle {{
            font-size: 16pt;
            font-weight: 600;
            color: #4f46e5; /* Indigo */
            margin: 0 0 40px 0;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        
        .cover-divider {{
            width: 80px;
            height: 4px;
            background-color: #4f46e5;
            margin-bottom: 40px;
            border-radius: 2px;
        }}
        
        .cover-description {{
            font-size: 11pt;
            color: #64748b;
            max-width: 500px;
            margin: 0 auto 80px auto;
        }}
        
        .cover-metadata {{
            font-size: 9pt;
            color: #94a3b8;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
            width: 300px;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: #0f172a;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            page-break-after: avoid;
            margin-top: 1.8em;
            margin-bottom: 0.6em;
        }}
        
        h1 {{
            font-size: 22pt;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 8px;
            color: #0f172a;
        }}
        
        h2 {{
            font-size: 15pt;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 6px;
            color: #1e293b;
        }}
        
        h3 {{
            font-size: 12pt;
            color: #334155;
        }}
        
        p {{
            margin-top: 0;
            margin-bottom: 1.2em;
            text-align: justify;
        }}
        
        /* Lists */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 1.2em;
            padding-left: 20px;
        }}
        
        li {{
            margin-bottom: 0.4em;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            page-break-inside: avoid;
            font-size: 9.5pt;
        }}
        
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 10px 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #334155;
        }}
        
        tr:nth-child(even) {{
            background-color: #fafbfd;
        }}
        
        /* Links */
        a {{
            color: #4f46e5;
            text-decoration: none;
            font-weight: 500;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* Code Blocks */
        code {{
            font-family: 'JetBrains Mono', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 9pt;
            font-weight: 500;
        }}
        
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 1.5em 0;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            font-size: 8.5pt;
            color: #334155;
            line-height: 1.5;
        }}
        
        /* Print helpers */
        .page-break {{
            page-break-before: always;
        }}
        
        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 2em 0;
        }}
    </style>
</head>
<body>

    <!-- Cover Page -->
    <div class="cover-page">
        <h1 class="cover-title">AI-Driven Customer Insights<br>& Recommendation System</h1>
        <h2 class="cover-subtitle">Interview Preparation Guide</h2>
        <div class="cover-divider"></div>
        <p class="cover-description">
            A premium, comprehensive resource covering end-to-end architecture, data flow diagrams, 
            critical logic defects (data leakage), and 200 tailored Q&As across ML, Software Engineering, 
            Web APIs, and System Design.
        </p>
        <div class="cover-metadata">
            <strong>Target Role:</strong> ML Engineer / Full-Stack Developer<br>
            <strong>Version:</strong> 1.0.0 (Production Ready)<br>
            <strong>Created:</strong> June 2026
        </div>
    </div>

    <!-- Main Content -->
    <div class="content-container">
        {html_body}
    </div>

</body>
</html>
"""

    # Write HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_document)

    print(f"Generated HTML successfully at {html_path}")

    # Use Google Chrome CLI to print the HTML to a PDF
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    
    # Run the headless print command
    print(f"Invoking Google Chrome headless to compile PDF...")
    command = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Success: PDF generated successfully at {pdf_path}")
    except Exception as e:
        print(f"Error during Chrome PDF print: {e}")

if __name__ == "__main__":
    md_file = "/Users/anshrathore/.gemini/antigravity-ide/brain/0489659a-0a3a-4543-aa19-40107be52fd7/interview_preparation_guide.md"
    html_file = "/Users/anshrathore/Desktop/AI-Driven-Customer-Insights-and-Recommendation-System/interview_preparation_guide.html"
    pdf_file = "/Users/anshrathore/Desktop/AI-Driven-Customer-Insights-and-Recommendation-System/interview_preparation_guide.pdf"
    build_beautiful_pdf(md_file, html_file, pdf_file)
