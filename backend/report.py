"""
report.py — PDF Weekly/Monthly Report Generator
================================================
Generates a beautiful PDF report with:
  - Weekly mood summary
  - Mental state breakdown chart (as text bars)
  - App usage statistics
  - Weekly letter
  - Personalised insights

Add this to backend/main.py with:
  from report import generate_pdf_report

Run standalone:
  python backend/report.py

Requires:
  pip install reportlab
"""

import os
import sys
import io
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'collector'))
sys.path.insert(0, os.path.join(ROOT, 'model'))

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.graphics.shapes import Drawing, Rect, String, Circle
    from reportlab.graphics import renderPDF
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ── Brand colors ──────────────────────────────────────────────────────────────
SAGE        = colors.HexColor('#7C9E87')
SAGE_DARK   = colors.HexColor('#4A7A5A')
PEACH       = colors.HexColor('#F2A07B')
CREAM       = colors.HexColor('#FDF8F2')
CHARCOAL    = colors.HexColor('#2C2C2C')
MUTED       = colors.HexColor('#8A8A8A')
LIGHT_BORDER= colors.HexColor('#E8E0D5')
WHITE       = colors.white

STATE_COLORS = {
    'Normal'              : colors.HexColor('#7C9E87'),
    'Anxiety'             : colors.HexColor('#E8845A'),
    'Stress'              : colors.HexColor('#D4A843'),
    'Depression'          : colors.HexColor('#7B8EC8'),
    'Bipolar'             : colors.HexColor('#C17BC0'),
    'Suicidal'            : colors.HexColor('#C05A5A'),
    'Personality Disorder': colors.HexColor('#E09060'),
}

STATE_EMOJI = {
    'Normal': '😊', 'Anxiety': '😰', 'Stress': '😤',
    'Depression': '😔', 'Bipolar': '🔄', 'Suicidal': '🆘',
    'Personality Disorder': '🌀',
}


def generate_pdf_report(period: str = 'week') -> bytes:
    """
    Generate a PDF report and return it as bytes.
    period: 'week' or 'month'
    """
    if not REPORTLAB_OK:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    from db import get_predictions, get_weekly_summaries, get_recent_diary_entries
    from predictor import generate_weekly_analysis

    # Load data
    days        = 7 if period == 'week' else 30
    predictions = get_predictions(days=days)
    summaries   = get_weekly_summaries()
    diary       = get_recent_diary_entries(days=days)
    weekly      = generate_weekly_analysis(predictions, summaries)

    # Build PDF in memory
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize    = A4,
        leftMargin  = 2.2*cm,
        rightMargin = 2.2*cm,
        topMargin   = 2*cm,
        bottomMargin= 2*cm,
    )

    styles  = _build_styles()
    content = []

    # ── Cover header ──────────────────────────────────────────────────────────
    content += _cover_header(period, styles)

    # ── Mental state summary ──────────────────────────────────────────────────
    content += _state_summary(predictions, styles)

    # ── Mood timeline ─────────────────────────────────────────────────────────
    content += _mood_timeline(predictions, styles)

    # ── Usage statistics ──────────────────────────────────────────────────────
    content += _usage_stats(summaries, styles)

    # ── Weekly letter ─────────────────────────────────────────────────────────
    if weekly.get('available'):
        content += _weekly_letter(weekly, styles)

    # ── Diary highlights ─────────────────────────────────────────────────────
    content += _diary_highlights(diary, styles)

    # ── Footer ───────────────────────────────────────────────────────────────
    content += _footer(styles)

    doc.build(content)
    return buffer.getvalue()


# ── Section builders ──────────────────────────────────────────────────────────

def _cover_header(period, styles):
    end_date   = date.today()
    start_date = end_date - timedelta(days=6 if period == 'week' else 29)
    date_range = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"

    elements = []

    # Top color bar
    drawing = Drawing(17*cm, 1.2*cm)
    drawing.add(Rect(0, 0, 17*cm, 1.2*cm, fillColor=SAGE_DARK, strokeColor=None))
    elements.append(drawing)
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph("Serenity", styles['app_name']))
    elements.append(Paragraph(
        f"{'Weekly' if period == 'week' else 'Monthly'} Mental Health Report",
        styles['report_title']
    ))
    elements.append(Paragraph(date_range, styles['date_range']))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BORDER))
    elements.append(Spacer(1, 0.4*cm))

    return elements


def _state_summary(predictions, styles):
    if not predictions:
        return [Paragraph("No prediction data available yet.", styles['body'])]

    elements = []
    elements.append(Paragraph("Mental State Overview", styles['section_title']))
    elements.append(Spacer(1, 0.3*cm))

    # Count states
    from collections import Counter
    state_counts = Counter(p['predicted_state'] for p in predictions)
    total        = len(predictions)
    dominant     = state_counts.most_common(1)[0][0]
    avg_conf     = sum(p.get('confidence', 0) for p in predictions) / max(total, 1)

    # Summary stats row
    summary_data = [
        ['Days Analysed', 'Dominant State', 'Avg Confidence', 'States Detected'],
        [
            str(total),
            dominant,
            f"{avg_conf:.0%}",
            str(len(state_counts)),
        ]
    ]
    t = Table(summary_data, colWidths=[4*cm, 4.5*cm, 4*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), SAGE),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0), 9),
        ('BACKGROUND',  (0,1), (-1,1), CREAM),
        ('FONTNAME',    (0,1), (-1,1), 'Helvetica'),
        ('FONTSIZE',    (0,1), (-1,1), 12),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CREAM]),
        ('GRID',        (0,0), (-1,-1), 0.5, LIGHT_BORDER),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [4]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4*cm))

    # State breakdown bars
    elements.append(Paragraph("State Breakdown", styles['sub_title']))
    elements.append(Spacer(1, 0.2*cm))

    bar_data = [['State', 'Days', 'Proportion']]
    for state, count in state_counts.most_common():
        pct     = count / total
        bar_len = int(pct * 30)
        bar     = '█' * bar_len + '░' * (30 - bar_len)
        bar_data.append([state, str(count), f"{bar}  {pct:.0%}"])

    bt = Table(bar_data, colWidths=[4.5*cm, 2*cm, 10.5*cm])
    bt.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#F5F0EA')),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('TEXTCOLOR',   (0,1), (0,-1), CHARCOAL),
        ('TEXTCOLOR',   (2,1), (2,-1), SAGE_DARK),
        ('FONTNAME',    (2,1), (2,-1), 'Courier'),
        ('ALIGN',       (1,0), (1,-1), 'CENTER'),
        ('GRID',        (0,0), (-1,-1), 0.3, LIGHT_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, CREAM]),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(bt)
    elements.append(Spacer(1, 0.5*cm))
    return elements


def _mood_timeline(predictions, styles):
    if not predictions:
        return []

    elements = []
    elements.append(Paragraph("Daily Mood Timeline", styles['section_title']))
    elements.append(Spacer(1, 0.3*cm))

    rows = [['Date', 'Day', 'State', 'Confidence', 'Data Source']]
    for p in reversed(predictions):
        d       = p['date'] if isinstance(p['date'], date) else date.fromisoformat(str(p['date']))
        tw      = p.get('text_weight', 0.5) or 0.5
        nw      = p.get('numeric_weight', 0.5) or 0.5
        source  = f"Text {tw:.0%} / Usage {nw:.0%}"
        rows.append([
            d.strftime('%d %b'),
            d.strftime('%a'),
            p['predicted_state'],
            f"{(p.get('confidence',0) or 0):.0%}",
            source,
        ])

    t = Table(rows, colWidths=[2.5*cm, 2*cm, 4.5*cm, 3*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), SAGE),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, CREAM]),
        ('GRID',        (0,0), (-1,-1), 0.3, LIGHT_BORDER),
        ('ALIGN',       (3,0), (3,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*cm))
    return elements


def _usage_stats(summaries, styles):
    if not summaries:
        return []

    elements = []
    elements.append(Paragraph("App Usage Statistics", styles['section_title']))
    elements.append(Spacer(1, 0.3*cm))

    def avg(key):
        vals = [s.get(key, 0) or 0 for s in summaries]
        return sum(vals) / max(len(vals), 1)

    stats = [
        ['Metric',               'Daily Average',                    'Insight'],
        ['Total Screen Time',    f"{avg('total_screen_time_mins'):.0f} mins",
         'High' if avg('total_screen_time_mins') > 360 else 'Moderate'],
        ['Social Media',         f"{avg('social_media_mins'):.0f} mins",
         'Watch this' if avg('social_media_mins') > 90 else 'Healthy'],
        ['Work / Study',         f"{avg('work_app_mins'):.0f} mins",    '-'],
        ['Active (typing)',       f"{avg('active_time_mins'):.0f} mins", '-'],
        ['Idle Time',            f"{avg('idle_time_mins'):.0f} mins",    '-'],
        ['Late Night Usage',     f"{avg('late_night_usage_mins'):.0f} mins",
         'Concerning' if avg('late_night_usage_mins') > 30 else 'OK'],
        ['Breaks Taken',         f"{avg('break_count'):.1f} /day",
         'Good' if avg('break_count') >= 3 else 'Take more breaks'],
        ['Keystrokes',           f"{avg('keystrokes_count'):.0f} /day",  '-'],
    ]

    t = Table(stats, colWidths=[5*cm, 4*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), SAGE),
        ('TEXTCOLOR',    (0,0), (-1,0), WHITE),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, CREAM]),
        ('GRID',         (0,0), (-1,-1), 0.3, LIGHT_BORDER),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*cm))
    return elements


def _weekly_letter(weekly, styles):
    elements = []
    elements.append(Paragraph("Your Weekly Letter", styles['section_title']))
    elements.append(Spacer(1, 0.3*cm))

    letter_box = [[Paragraph(weekly['weekly_letter'].replace('\n', '<br/>'), styles['letter'])]]
    lt = Table(letter_box, colWidths=[17*cm])
    lt.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), CREAM),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING',(0,0), (-1,-1), 16),
        ('TOPPADDING',  (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0),(-1,-1), 14),
        ('LINEAFTER',   (0,0), (0,-1), 3, SAGE),
    ]))
    elements.append(lt)
    elements.append(Spacer(1, 0.5*cm))
    return elements


def _diary_highlights(diary, styles):
    if not diary:
        return []

    elements = []
    elements.append(Paragraph("Diary Highlights", styles['section_title']))
    elements.append(Paragraph(
        "A snapshot of your recent entries (first 120 characters each).",
        styles['caption']
    ))
    elements.append(Spacer(1, 0.3*cm))

    for entry in diary[:5]:
        d       = entry['date'] if isinstance(entry['date'], date) else date.fromisoformat(str(entry['date']))
        snippet = str(entry['entry_text'])[:120] + ('...' if len(str(entry['entry_text'])) > 120 else '')
        row = [[
            Paragraph(d.strftime('%d %b %Y'), styles['diary_date']),
            Paragraph(f'"{snippet}"', styles['diary_snippet']),
        ]]
        t = Table(row, colWidths=[3*cm, 14*cm])
        t.setStyle(TableStyle([
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING',  (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ('LINEBELOW',   (0,0), (-1,-1), 0.3, LIGHT_BORDER),
        ]))
        elements.append(t)

    elements.append(Spacer(1, 0.4*cm))
    return elements


def _footer(styles):
    elements = []
    elements.append(HRFlowable(width="100%", thickness=1, color=SAGE))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        f"Generated by Serenity &nbsp;|&nbsp; {datetime.now().strftime('%d %B %Y at %H:%M')} &nbsp;|&nbsp; "
        "Your data is private and stored locally.",
        styles['footer']
    ))
    return elements


# ── Style definitions ─────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    def ps(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    return {
        'app_name': ps('app_name',
            fontName='Helvetica-Bold', fontSize=11,
            textColor=SAGE_DARK, spaceAfter=2),

        'report_title': ps('report_title',
            fontName='Helvetica-Bold', fontSize=22,
            textColor=CHARCOAL, spaceAfter=4),

        'date_range': ps('date_range',
            fontName='Helvetica', fontSize=11,
            textColor=MUTED, spaceAfter=6),

        'section_title': ps('section_title',
            fontName='Helvetica-Bold', fontSize=13,
            textColor=SAGE_DARK, spaceBefore=8, spaceAfter=4),

        'sub_title': ps('sub_title',
            fontName='Helvetica-Bold', fontSize=10,
            textColor=CHARCOAL, spaceAfter=3),

        'body': ps('body',
            fontName='Helvetica', fontSize=9,
            textColor=CHARCOAL, leading=14, spaceAfter=4),

        'letter': ps('letter',
            fontName='Helvetica', fontSize=10,
            textColor=CHARCOAL, leading=16),

        'caption': ps('caption',
            fontName='Helvetica-Oblique', fontSize=8,
            textColor=MUTED, spaceAfter=3),

        'diary_date': ps('diary_date',
            fontName='Helvetica-Bold', fontSize=8,
            textColor=SAGE_DARK),

        'diary_snippet': ps('diary_snippet',
            fontName='Helvetica-Oblique', fontSize=9,
            textColor=CHARCOAL, leading=13),

        'footer': ps('footer',
            fontName='Helvetica', fontSize=7,
            textColor=MUTED, alignment=TA_CENTER),
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not REPORTLAB_OK:
        print("Install reportlab first: pip install reportlab")
        sys.exit(1)

    print("Generating PDF report...")
    pdf_bytes = generate_pdf_report('week')
    out_path  = os.path.join(ROOT, 'data', 'serenity_report.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(pdf_bytes)
    print(f"Report saved to: {out_path}")
