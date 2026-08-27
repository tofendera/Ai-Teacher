from pathlib import Path
import io
import json
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openai import OpenAI

from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = ROOT_DIR / "static"

FONT_REGULAR = BASE_DIR / "THSarabun.ttf"
FONT_BOLD = BASE_DIR / "THSarabunBold.ttf"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.5",
)

STATIC_DIR.mkdir(exist_ok=True)


# =========================================================
# FONT
# =========================================================

FONT_ERROR = None

try:
    if not FONT_REGULAR.exists():
        FONT_ERROR = f"ไม่พบไฟล์ Font: {FONT_REGULAR}"

    elif not FONT_BOLD.exists():
        FONT_ERROR = f"ไม่พบไฟล์ Font: {FONT_BOLD}"

    else:
        pdfmetrics.registerFont(
            TTFont(
                "THSarabun",
                str(FONT_REGULAR),
            )
        )

        pdfmetrics.registerFont(
            TTFont(
                "THSarabunBold",
                str(FONT_BOLD),
            )
        )

except Exception as e:
    FONT_ERROR = str(e)


# =========================================================
# OPENAI
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


# =========================================================
# MODELS
# =========================================================

class GenerateRequest(BaseModel):
    prompt: str
    teacher_name: str = ""
    question_count: int = Field(default=10, ge=1, le=50)
    question_types: list[str] = Field(default_factory=lambda: ["multiple_choice"])
    difficulty: str = "mixed"


class PDFRequest(BaseModel):
    data: dict[str, Any]


# =========================================================
# BASIC HELPERS
# =========================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


def safe_text(value: Any, fallback: str = "") -> str:
    text = clean_text(value)
    return text if text else fallback


def normalize_list(value: Any) -> list:
    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [value]


def strip_markdown(text: str) -> str:
    text = clean_text(text)

    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.replace("**", "")
    text = text.replace("__", "")

    return text.strip()


def escape_pdf(text: Any) -> str:
    """
    Escape text สำหรับ ReportLab Paragraph
    """
    text = safe_text(text)

    text = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return text


def question_type_name(value: str) -> str:
    names = {
        "multiple_choice": "ปรนัย",
        "fill_blank": "เติมคำ",
        "calculation": "คำนวณ",
        "application": "ประยุกต์ใช้",
    }

    return names.get(
        value,
        value or "ข้อสอบ",
    )


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(text: str) -> dict:
    text = clean_text(text)

    # ลบ ```json
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # หา object แรก
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(
        "AI ส่งข้อมูลกลับมาในรูปแบบที่ไม่ถูกต้อง"
    )


# =========================================================
# AI GENERATION
# =========================================================

SYSTEM_PROMPT = """
คุณเป็นผู้เชี่ยวชาญด้านการออกแบบการเรียนรู้สำหรับโรงเรียนไทย

หน้าที่คือสร้างชุดการสอนที่ครูสามารถนำไปใช้ได้จริง

ต้องตอบกลับเป็น JSON เท่านั้น
ห้ามใส่ Markdown
ห้ามใส่ ```json
ห้ามมีข้อความอธิบายนอก JSON

รูปแบบ JSON ต้องเป็น:

{
  "summary": {
    "grade": "...",
    "subject": "...",
    "topic": "...",
    "duration": "..."
  },

  "lesson_plan": {
    "objective": [
      "...",
      "...",
      "..."
    ],
    "content": "...",
    "key_points": [
      "...",
      "..."
    ],
    "examples": [
      "...",
      "..."
    ],
    "steps": [
      {
        "time": "...",
        "title": "...",
        "detail": "..."
      }
    ],
    "assessment": "..."
  },

  "worksheet": [
    {
      "no": 1,
      "question": "...",
      "answer": "..."
    }
  ],

  "quiz": [
    {
      "no": 1,
      "type": "multiple_choice",
      "question": "...",
      "options": [
        "...",
        "...",
        "...",
        "..."
      ],
      "answer": "...",
      "explanation": "..."
    }
  ]
}

กฎสำคัญ:

1. ภาษาไทยต้องอ่านเป็นธรรมชาติ
2. ข้อสอบต้องเหมาะกับระดับชั้น
3. ห้ามสร้างคำถามที่คลุมเครือ
4. เฉลยต้องถูกต้อง
5. ถ้าเป็นปรนัย ให้มีตัวเลือก 4 ตัวเลือก
6. ตัวเลือกต้องมีเนื้อหาจริง ไม่ใช่ ก ข ค ง เฉย ๆ
7. ถ้าเป็นเติมคำ ไม่ต้องใส่ options
8. ถ้าเป็นคำนวณ ต้องมีโจทย์และคำตอบ
9. ถ้าเป็นประยุกต์ใช้ ต้องเป็นสถานการณ์ที่นักเรียนคิดวิเคราะห์
10. worksheet ต้องมีจำนวนตาม question_count
11. quiz ต้องมีจำนวนตาม question_count
12. ห้ามใช้ Emoji ในเนื้อหาเอกสารราชการ
13. ชื่อครูให้ใช้ตามที่ผู้ใช้ส่งมา
14. ถ้าไม่ระบุชื่อครู ให้ใช้ "ไม่ระบุ"
15. summary ต้องพยายามวิเคราะห์ วิชา ระดับชั้น เรื่อง และเวลา จาก prompt
16. lesson_plan ต้องมีเนื้อหาเพียงพอสำหรับครูนำไปสอนได้จริง
17. ขั้นตอนการสอนต้องเรียงตามเวลา
"""


def generate_with_ai(request: GenerateRequest) -> dict:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render"
        )

    teacher_name = (
        request.teacher_name.strip()
        if request.teacher_name
        else "ไม่ระบุ"
    )

    user_prompt = f"""
สร้างชุดการสอนจากข้อมูลต่อไปนี้

หัวข้อ/คำสั่ง:
{request.prompt}

ชื่อครู:
{teacher_name}

จำนวนข้อ:
{request.question_count}

รูปแบบข้อสอบ:
{", ".join(request.question_types)}

ระดับความยาก:
{request.difficulty}

โปรดสร้างเนื้อหาที่เหมาะกับนักเรียนไทย
และจัดข้อมูลให้พร้อมสำหรับทำเป็นเอกสารการสอนและใบงานจริง
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.5,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "AI ไม่ได้ส่งข้อมูลกลับมา"
            )

        data = extract_json(content)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI สร้างชุดการสอนไม่สำเร็จ: {e}"
        )

    # =====================================================
    # NORMALIZE
    # =====================================================

    summary = data.get("summary") or {}

    summary["grade"] = safe_text(
        summary.get("grade"),
        "ไม่ระบุ",
    )

    summary["subject"] = safe_text(
        summary.get("subject"),
        "ไม่ระบุ",
    )

    summary["topic"] = safe_text(
        summary.get("topic"),
        request.prompt,
    )

    summary["duration"] = safe_text(
        summary.get("duration"),
        "ไม่ระบุ",
    )

    data["summary"] = summary

    lesson = data.get("lesson_plan") or {}

    lesson["objective"] = normalize_list(
        lesson.get("objective")
    )

    lesson["content"] = safe_text(
        lesson.get("content")
    )

    lesson["key_points"] = normalize_list(
        lesson.get("key_points")
    )

    lesson["examples"] = normalize_list(
        lesson.get("examples")
    )

    lesson["steps"] = normalize_list(
        lesson.get("steps")
    )

    lesson["assessment"] = safe_text(
        lesson.get("assessment")
    )

    data["lesson_plan"] = lesson

    worksheet = normalize_list(
        data.get("worksheet")
    )

    normalized_worksheet = []

    for index, item in enumerate(
        worksheet,
        start=1,
    ):
        if isinstance(item, dict):
            normalized_worksheet.append(
                {
                    "no": index,
                    "question": safe_text(
                        item.get("question")
                    ),
                    "answer": safe_text(
                        item.get("answer")
                    ),
                }
            )

    data["worksheet"] = normalized_worksheet

    quiz = normalize_list(
        data.get("quiz")
    )

    normalized_quiz = []

    for index, item in enumerate(
        quiz,
        start=1,
    ):
        if isinstance(item, dict):
            normalized_quiz.append(
                {
                    "no": index,
                    "type": safe_text(
                        item.get("type"),
                        "multiple_choice",
                    ),
                    "question": safe_text(
                        item.get("question")
                    ),
                    "options": normalize_list(
                        item.get("options")
                    ),
                    "answer": safe_text(
                        item.get("answer")
                    ),
                    "explanation": safe_text(
                        item.get("explanation")
                    ),
                }
            )

    data["quiz"] = normalized_quiz

    data["teacher_name"] = teacher_name

    return data


# =========================================================
# PDF STYLES
# =========================================================

def create_pdf_styles():
    if FONT_ERROR:
        raise RuntimeError(
            f"ฟอนต์ไม่พร้อมใช้งาน: {FONT_ERROR}"
        )

    return {
        "title": ParagraphStyle(
            "PDFTitle",
            fontName="THSarabunBold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
        ),

        "subtitle": ParagraphStyle(
            "PDFSubtitle",
            fontName="THSarabun",
            fontSize=18,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),

        "meta": ParagraphStyle(
            "PDFMeta",
            fontName="THSarabun",
            fontSize=16,
            leading=20,
            alignment=TA_LEFT,
            spaceAfter=7 * mm,
        ),

        "heading": ParagraphStyle(
            "PDFHeading",
            fontName="THSarabunBold",
            fontSize=20,
            leading=25,
            alignment=TA_LEFT,
            spaceBefore=5 * mm,
            spaceAfter=4 * mm,
        ),

        "subheading": ParagraphStyle(
            "PDFSubHeading",
            fontName="THSarabunBold",
            fontSize=18,
            leading=23,
            alignment=TA_LEFT,
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
        ),

        "body": ParagraphStyle(
            "PDFBody",
            fontName="THSarabun",
            fontSize=16,
            leading=23,
            alignment=TA_LEFT,
            firstLineIndent=8 * mm,
            spaceAfter=3 * mm,
        ),

        "body_no_indent": ParagraphStyle(
            "PDFBodyNoIndent",
            fontName="THSarabun",
            fontSize=16,
            leading=23,
            alignment=TA_LEFT,
            firstLineIndent=0,
            spaceAfter=3 * mm,
        ),

        "bullet": ParagraphStyle(
            "PDFBullet",
            fontName="THSarabun",
            fontSize=16,
            leading=23,
            alignment=TA_LEFT,
            leftIndent=8 * mm,
            firstLineIndent=-5 * mm,
            spaceAfter=2 * mm,
        ),

        "question": ParagraphStyle(
            "PDFQuestion",
            fontName="THSarabun",
            fontSize=17,
            leading=24,
            alignment=TA_LEFT,
            firstLineIndent=8 * mm,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),

        "answer": ParagraphStyle(
            "PDFAnswer",
            fontName="THSarabun",
            fontSize=16,
            leading=22,
            alignment=TA_LEFT,
            firstLineIndent=8 * mm,
            spaceAfter=5 * mm,
        ),

        "option": ParagraphStyle(
            "PDFOption",
            fontName="THSarabun",
            fontSize=16,
            leading=22,
            alignment=TA_LEFT,
            leftIndent=15 * mm,
            firstLineIndent=0,
            spaceAfter=1.5 * mm,
        ),

        "step_title": ParagraphStyle(
            "PDFStepTitle",
            fontName="THSarabunBold",
            fontSize=17,
            leading=22,
            alignment=TA_LEFT,
            spaceAfter=1 * mm,
        ),

        "step_detail": ParagraphStyle(
            "PDFStepDetail",
            fontName="THSarabun",
            fontSize=16,
            leading=22,
            alignment=TA_LEFT,
            firstLineIndent=8 * mm,
            spaceAfter=3 * mm,
        ),

        "small": ParagraphStyle(
            "PDFSmall",
            fontName="THSarabun",
            fontSize=14,
            leading=19,
            alignment=TA_LEFT,
        ),
    }


# =========================================================
# PDF HELPERS
# =========================================================

def paragraph_text(value: Any) -> str:
    text = strip_markdown(
        safe_text(value)
    )

    # แปลง newline ให้เป็น line break
    text = escape_pdf(text)
    text = text.replace(
        "\n",
        "<br/>"
    )

    return text


def add_body(
    story: list,
    text: Any,
    styles: dict,
    indent: bool = True,
):
    text = clean_text(text)

    if not text:
        return

    style = (
        styles["body"]
        if indent
        else styles["body_no_indent"]
    )

    # แยกย่อหน้าตาม newline
    parts = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    for part in parts:
        story.append(
            Paragraph(
                paragraph_text(part),
                style,
            )
        )


def add_bullets(
    story: list,
    items: list,
    styles: dict,
):
    for item in items:
        text = clean_text(item)

        if not text:
            continue

        story.append(
            Paragraph(
                "• " + paragraph_text(text),
                styles["bullet"],
            )
        )


def add_heading(
    story: list,
    number: str,
    title: str,
    styles: dict,
):
    story.append(
        Paragraph(
            escape_pdf(
                f"{number}. {title}"
            ),
            styles["heading"],
        )
    )


# =========================================================
# PDF CREATION
# =========================================================

def build_pdf(data: dict) -> bytes:
    styles = create_pdf_styles()

    buffer = io.BytesIO()

    # สำคัญ:
    # ห้ามใช้ภาษาไทยใน metadata
    # เพราะ ReportLab บางส่วนใช้ latin-1
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,

        # ASCII เท่านั้น ป้องกัน latin-1 error
        title="Teacher Pack",
        author="Teacher Pack",
        subject="Lesson Plan",
        creator="Teacher Pack",
    )

    story = []

    summary = data.get("summary") or {}
    lesson = data.get("lesson_plan") or {}

    teacher_name = safe_text(
        data.get("teacher_name"),
        "ไม่ระบุ",
    )

    grade = safe_text(
        summary.get("grade"),
        "ไม่ระบุ",
    )

    subject = safe_text(
        summary.get("subject"),
        "ไม่ระบุ",
    )

    topic = safe_text(
        summary.get("topic"),
        "ไม่ระบุ",
    )

    duration = safe_text(
        summary.get("duration"),
        "ไม่ระบุ",
    )

    # =====================================================
    # 1. LESSON PLAN
    # =====================================================

    story.append(
        Paragraph(
            "แผนการจัดการเรียนรู้",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {escape_pdf(topic)}",
            styles["subtitle"],
        )
    )

    # ข้อมูล 4 ช่องในบรรทัดเดียว
    meta_line = (
        f"วิชา {escape_pdf(subject)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ระดับชั้น {escape_pdf(grade)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"เวลา {escape_pdf(duration)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ครูผู้สอน {escape_pdf(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta_line,
            styles["meta"],
        )
    )

    # จุดประสงค์
    add_heading(
        story,
        "1",
        "จุดประสงค์การเรียนรู้",
        styles,
    )

    add_bullets(
        story,
        lesson.get("objective", []),
        styles,
    )

    # เนื้อหา
    add_heading(
        story,
        "2",
        "เนื้อหาที่ใช้สอน",
        styles,
    )

    add_body(
        story,
        lesson.get("content"),
        styles,
        indent=True,
    )

    # สาระสำคัญ
    if lesson.get("key_points"):
        story.append(
            Paragraph(
                "สาระสำคัญ",
                styles["subheading"],
            )
        )

        add_bullets(
            story,
            lesson.get("key_points", []),
            styles,
        )

    # ตัวอย่าง
    if lesson.get("examples"):
        story.append(
            Paragraph(
                "ตัวอย่างสำหรับใช้สอน",
                styles["subheading"],
            )
        )

        add_bullets(
            story,
            lesson.get("examples", []),
            styles,
        )

    # ขั้นตอนการสอน
    if lesson.get("steps"):
        story.append(
            Paragraph(
                "3. ขั้นตอนการจัดการเรียนรู้",
                styles["heading"],
            )
        )

        for step in lesson.get("steps", []):
            if not isinstance(step, dict):
                continue

            time = safe_text(
                step.get("time")
            )

            title = safe_text(
                step.get("title")
            )

            detail = safe_text(
                step.get("detail")
            )

            title_text = (
                f"{time} {title}"
                if time
                else title
            )

            block = [
                Paragraph(
                    escape_pdf(title_text),
                    styles["step_title"],
                )
            ]

            if detail:
                block.append(
                    Paragraph(
                        paragraph_text(detail),
                        styles["step_detail"],
                    )
                )

            story.append(
                KeepTogether(block)
            )

    # การประเมินผล
    if lesson.get("assessment"):
        story.append(
            Paragraph(
                "4. การประเมินผล",
                styles["heading"],
            )
        )

        add_body(
            story,
            lesson.get("assessment"),
            styles,
            indent=True,
        )

    # =====================================================
    # 2. WORKSHEET
    # =====================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "ใบงาน",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {escape_pdf(topic)}",
            styles["subtitle"],
        )
    )

    meta_line_worksheet = (
        f"วิชา {escape_pdf(subject)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ระดับชั้น {escape_pdf(grade)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"เวลา {escape_pdf(duration)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ครูผู้สอน {escape_pdf(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta_line_worksheet,
            styles["meta"],
        )
    )

    story.append(
        Paragraph(
            "ชื่อ-สกุล ................................................................................................................",
            styles["body_no_indent"],
        )
    )

    story.append(
        Paragraph(
            "ชั้น .................... เลขที่ .................... วันที่ ....................",
            styles["body_no_indent"],
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "คำชี้แจง",
            styles["subheading"],
        )
    )

    story.append(
        Paragraph(
            "ให้นักเรียนอ่านคำถามแต่ละข้อ และเขียนคำตอบลงในพื้นที่ที่กำหนด",
            styles["body_no_indent"],
        )
    )

    worksheet = data.get("worksheet") or []

    for item in worksheet:
        if not isinstance(item, dict):
            continue

        no = item.get("no", "")
        question = clean_text(
            item.get("question")
        )

        if not question:
            continue

        # ---------------------------------------------
        # รูปแบบข้อแบบที่ผู้ใช้ต้องการ
        #
        # ข้อ 1  คำถาม...
        #
        # คำตอบ........................................
        # ---------------------------------------------

        q_text = (
            f"ข้อ {no}  {question}"
        )

        story.append(
            Paragraph(
                paragraph_text(q_text),
                styles["question"],
            )
        )

        story.append(
            Paragraph(
                "คำตอบ ................................................................................................................",
                styles["answer"],
            )
        )

        # เว้นระยะก่อนข้อถัดไป
        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

    # =====================================================
    # 3. QUIZ
    # =====================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "แบบทดสอบ",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {escape_pdf(topic)}",
            styles["subtitle"],
        )
    )

    meta_line_quiz = (
        f"วิชา {escape_pdf(subject)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ระดับชั้น {escape_pdf(grade)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"เวลา {escape_pdf(duration)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ครูผู้สอน {escape_pdf(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta_line_quiz,
            styles["meta"],
        )
    )

    story.append(
        Paragraph(
            "ชื่อ-สกุล ................................................................................................................",
            styles["body_no_indent"],
        )
    )

    story.append(
        Paragraph(
            "ชั้น .................... เลขที่ .................... วันที่ ....................",
            styles["body_no_indent"],
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "คำชี้แจง",
            styles["subheading"],
        )
    )

    story.append(
        Paragraph(
            "ให้นักเรียนเลือกหรือเขียนคำตอบที่ถูกต้องที่สุด",
            styles["body_no_indent"],
        )
    )

    quiz = data.get("quiz") or []

    for item in quiz:
        if not isinstance(item, dict):
            continue

        no = item.get("no", "")
        qtype = question_type_name(
            safe_text(
                item.get("type"),
                "multiple_choice",
            )
        )

        question = clean_text(
            item.get("question")
        )

        if not question:
            continue

        story.append(
            Paragraph(
                paragraph_text(
                    f"ข้อ {no}  ({qtype})"
                ),
                styles["subheading"],
            )
        )

        story.append(
            Paragraph(
                paragraph_text(question),
                styles["question"],
            )
        )

        options = item.get("options") or []

        letters = [
            "ก.",
            "ข.",
            "ค.",
            "ง.",
        ]

        for index, option in enumerate(options):
            if index >= len(letters):
                break

            story.append(
                Paragraph(
                    paragraph_text(
                        f"{letters[index]} {option}"
                    ),
                    styles["option"],
                )
            )

        if not options:
            story.append(
                Paragraph(
                    "คำตอบ ................................................................................................................",
                    styles["answer"],
                )
            )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

    # =====================================================
    # 4. ANSWER KEY
    # =====================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "เฉลย",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {escape_pdf(topic)}",
            styles["subtitle"],
        )
    )

    meta_line_answer = (
        f"วิชา {escape_pdf(subject)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ระดับชั้น {escape_pdf(grade)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"เวลา {escape_pdf(duration)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ครูผู้สอน {escape_pdf(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta_line_answer,
            styles["meta"],
        )
    )

    story.append(
        Paragraph(
            "เฉลยใบงาน",
            styles["heading"],
        )
    )

    for item in worksheet:
        if not isinstance(item, dict):
            continue

        no = item.get("no", "")
        answer = safe_text(
            item.get("answer"),
            "ไม่ระบุ",
        )

        story.append(
            Paragraph(
                paragraph_text(
                    f"ข้อ {no}  {answer}"
                ),
                styles["body_no_indent"],
            )
        )

    story.append(
        Paragraph(
            "เฉลยแบบทดสอบ",
            styles["heading"],
        )
    )

    for item in quiz:
        if not isinstance(item, dict):
            continue

        no = item.get("no", "")
        answer = safe_text(
            item.get("answer"),
            "ไม่ระบุ",
        )

        explanation = safe_text(
            item.get("explanation")
        )

        story.append(
            Paragraph(
                paragraph_text(
                    f"ข้อ {no}  {answer}"
                ),
                styles["body_no_indent"],
            )
        )

        if explanation:
            story.append(
                Paragraph(
                    paragraph_text(
                        explanation
                    ),
                    styles["body"],
                )
            )

    # =====================================================
    # BUILD
    # =====================================================

    try:
        doc.build(story)
    except Exception as e:
        raise RuntimeError(
            f"สร้าง PDF ไม่สำเร็จ: {e}"
        )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# ROUTES
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            "<h1>ไม่พบ static/index.html</h1>",
            status_code=500,
        )

    return HTMLResponse(
        index_file.read_text(
            encoding="utf-8"
        )
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.5",
        "font_regular": FONT_REGULAR.name,
        "font_bold": FONT_BOLD.name,
        "font_error": FONT_ERROR,
        "openai_configured": bool(
            OPENAI_API_KEY
        ),
    }


@app.post("/api/generate")
async def api_generate(
    request: GenerateRequest,
):
    return generate_with_ai(request)


@app.post("/api/pdf")
async def api_pdf(
    request: PDFRequest,
):
    try:
        pdf_bytes = build_pdf(
            request.data
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                'attachment; filename="teacher_pack.pdf"'
            )
        },
    )


# =========================================================
# STATIC
# =========================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static",
)
