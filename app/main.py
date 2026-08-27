from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Any
import os
import json
import re
import io
import traceback

from openai import OpenAI

from reportlab.lib import colors
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
    KeepTogether,
    PageBreak,
)
from reportlab.pdfgen import canvas


# =========================================================
# PATH
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STATIC_DIR = PROJECT_DIR / "static"

FONT_REGULAR = APP_DIR / "THSarabun.ttf"
FONT_BOLD = APP_DIR / "THSarabunBold.ttf"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.8.0",
)


# =========================================================
# FONT
# =========================================================

FONT_READY = False
FONT_ERROR = None

try:

    if not FONT_REGULAR.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_REGULAR}"
        )

    if not FONT_BOLD.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_BOLD}"
        )

    pdfmetrics.registerFont(
        TTFont(
            "THSarabun",
            str(FONT_REGULAR)
        )
    )

    pdfmetrics.registerFont(
        TTFont(
            "THSarabunBold",
            str(FONT_BOLD)
        )
    )

    FONT_READY = True

except Exception as e:

    FONT_ERROR = str(e)
    FONT_READY = False


# =========================================================
# OPENAI
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


# =========================================================
# STATIC
# =========================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# =========================================================
# MODELS
# =========================================================

class GenerateRequest(BaseModel):

    prompt: str

    teacher_name: str = "ไม่ได้ระบุ"

    question_count: int = 10

    question_types: List[str] = [
        "multiple_choice"
    ]

    difficulty: str = "mixed"


class PDFRequest(BaseModel):

    data: dict


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "app": "Teacher Pack",
        "version": "1.8.0",

        "font_regular": FONT_REGULAR.name,
        "font_bold": FONT_BOLD.name,

        "font_regular_exists": FONT_REGULAR.exists(),
        "font_bold_exists": FONT_BOLD.exists(),

        "font_ready": FONT_READY,
        "font_error": FONT_ERROR,

        "openai_configured": bool(OPENAI_API_KEY),
    }


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            """
            <h1>Teacher Pack</h1>
            <p>ไม่พบ static/index.html</p>
            """,
            status_code=500
        )

    return HTMLResponse(
        index_file.read_text(
            encoding="utf-8"
        )
    )


# =========================================================
# TEXT HELPERS
# =========================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    return text.strip()


def normalize_spaces(text: str) -> str:

    text = clean_text(text)

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def safe_int(value, default=0):

    try:
        return int(value)
    except Exception:
        return default


def strip_markdown(text: str) -> str:

    text = clean_text(text)

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text
    )

    text = re.sub(
        r"`(.*?)`",
        r"\1",
        text
    )

    return text


# =========================================================
# HTML ESCAPE FOR PDF
# =========================================================

def pdf_escape(text: str) -> str:

    text = clean_text(text)

    text = text.replace(
        "&",
        "&amp;"
    )

    text = text.replace(
        "<",
        "&lt;"
    )

    text = text.replace(
        ">",
        "&gt;"
    )

    return text


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(text: str):

    text = clean_text(text)

    # Remove markdown fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(
        "AI ส่งข้อมูลกลับมาไม่ใช่ JSON ที่ถูกต้อง"
    )


# =========================================================
# DEFAULT STRUCTURE
# =========================================================

def ensure_structure(data: dict):

    if not isinstance(data, dict):
        data = {}

    summary = data.get(
        "summary",
        {}
    )

    if not isinstance(summary, dict):
        summary = {}

    lesson_plan = data.get(
        "lesson_plan",
        {}
    )

    if not isinstance(lesson_plan, dict):
        lesson_plan = {}

    objective = lesson_plan.get(
        "objective",
        []
    )

    if not isinstance(objective, list):
        objective = [str(objective)]

    steps = lesson_plan.get(
        "steps",
        []
    )

    if not isinstance(steps, list):
        steps = []

    worksheet = data.get(
        "worksheet",
        []
    )

    if not isinstance(worksheet, list):
        worksheet = []

    quiz = data.get(
        "quiz",
        []
    )

    if not isinstance(quiz, list):
        quiz = []

    data["summary"] = summary
    data["lesson_plan"] = lesson_plan
    data["lesson_plan"]["objective"] = objective
    data["lesson_plan"]["steps"] = steps
    data["worksheet"] = worksheet
    data["quiz"] = quiz

    return data


# =========================================================
# AI GENERATION
# =========================================================

def generate_with_ai(request: GenerateRequest):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render"
        )

    prompt = clean_text(
        request.prompt
    )

    teacher_name = clean_text(
        request.teacher_name
    ) or "ไม่ได้ระบุ"

    question_count = max(
        1,
        min(
            int(request.question_count),
            50
        )
    )

    question_types = request.question_types

    difficulty = request.difficulty

    system_prompt = """
คุณคือผู้ช่วยจัดทำเอกสารการเรียนการสอนสำหรับครูในประเทศไทย

หน้าที่คือสร้างชุดการสอนที่สามารถนำไปใช้จริงได้

ต้องตอบกลับเป็น JSON เท่านั้น
ห้ามใส่ Markdown
ห้ามใส่ ```json
ห้ามใส่คำอธิบายก่อนหรือหลัง JSON

เนื้อหาต้องเหมาะสมกับระดับชั้น
ภาษาไทยต้องเป็นธรรมชาติ
จัดลำดับเนื้อหาให้เป็นเอกสารการเรียนการสอนจริง
อย่าเขียนเป็นข้อความยาวติดกัน
"""

    user_prompt = f"""
หัวข้อที่ครูต้องการสอน:
{prompt}

ชื่อครู:
{teacher_name}

จำนวนข้อสอบ:
{question_count}

รูปแบบข้อสอบ:
{", ".join(question_types)}

ระดับความยาก:
{difficulty}

สร้าง JSON ตามโครงสร้างนี้:

{{
  "summary": {{
    "grade": "ระดับชั้น",
    "subject": "วิชา",
    "topic": "หัวข้อ",
    "duration": "เวลา"
  }},

  "lesson_plan": {{
    "objective": [
      "จุดประสงค์ข้อ 1",
      "จุดประสงค์ข้อ 2",
      "จุดประสงค์ข้อ 3"
    ],

    "content": "เนื้อหาที่ใช้สอน",

    "important_points": [
      "สาระสำคัญข้อ 1",
      "สาระสำคัญข้อ 2"
    ],

    "examples": [
      "ตัวอย่างที่ใช้สอนข้อ 1",
      "ตัวอย่างที่ใช้สอนข้อ 2"
    ],

    "steps": [
      {{
        "time": "10 นาที",
        "title": "นำเข้าสู่บทเรียน",
        "detail": "รายละเอียดกิจกรรม"
      }},
      {{
        "time": "15 นาที",
        "title": "อธิบายเนื้อหา",
        "detail": "รายละเอียดกิจกรรม"
      }},
      {{
        "time": "20 นาที",
        "title": "ฝึกปฏิบัติ",
        "detail": "รายละเอียดกิจกรรม"
      }},
      {{
        "time": "10 นาที",
        "title": "สรุป",
        "detail": "รายละเอียดกิจกรรม"
      }}
    ],

    "assessment": "วิธีประเมินผล"
  }},

  "worksheet": [
    {{
      "no": 1,
      "question": "คำถาม",
      "answer": "คำตอบ"
    }}
  ],

  "quiz": [
    {{
      "no": 1,
      "type": "multiple_choice",
      "question": "คำถาม",
      "options": [
        "ตัวเลือก ก",
        "ตัวเลือก ข",
        "ตัวเลือก ค",
        "ตัวเลือก ง"
      ],
      "answer": "ก",
      "explanation": "คำอธิบาย"
    }}
  ]
}}

ข้อกำหนดสำคัญ:

1. worksheet ต้องมีจำนวนตามที่กำหนด
2. quiz ต้องมีจำนวนตามที่กำหนด
3. ตัวเลขข้อ no ต้องเริ่มจาก 1 และเรียง 1,2,3...
4. ถ้าเป็นปรนัยต้องมี options 4 ตัวเลือก
5. ถ้าเป็นเติมคำไม่ต้องมี options
6. ถ้าเป็นคำนวณต้องมีโจทย์ที่คำนวณได้จริง
7. ถ้าเป็นประยุกต์ใช้ต้องเป็นสถานการณ์ที่เหมาะกับเด็ก
8. อย่าใส่เลขข้อซ้ำในข้อความ question เช่นไม่ต้องเขียน "ข้อ 1" ใน question
9. ใช้ภาษาไทยที่เหมาะสมกับนักเรียน
10. worksheet ต้องเหมาะสำหรับพิมพ์แจกนักเรียน
"""

    try:

        response = client.chat.completions.create(

            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            ),

            temperature=0.7,

            response_format={
                "type": "json_object"
            },

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        raw = response.choices[0].message.content

        data = extract_json(
            raw
        )

        data = ensure_structure(
            data
        )

        data["teacher_name"] = teacher_name

        return data

    except Exception as e:

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"AI สร้างข้อมูลไม่สำเร็จ: {str(e)}"
        )


# =========================================================
# GENERATE API
# =========================================================

@app.post("/api/generate")
def generate(request: GenerateRequest):

    if not request.prompt.strip():

        raise HTTPException(
            status_code=400,
            detail="กรุณาระบุหัวข้อที่ต้องการสอน"
        )

    if not request.question_types:

        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ"
        )

    data = generate_with_ai(
        request
    )

    return data


# =========================================================
# PDF PAGE NUMBER
# =========================================================

class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):

        canvas.Canvas.__init__(
            self,
            *args,
            **kwargs
        )

        self.pages = []

    def showPage(self):

        self.pages.append(
            dict(self.__dict__)
        )

        self._startPage()

    def save(self):

        page_count = len(
            self.pages
        )

        for page in self.pages:

            self.__dict__.update(
                page
            )

            self.draw_page_number(
                page_count
            )

            canvas.Canvas.showPage(
                self
            )

        canvas.Canvas.save(
            self
        )

    def draw_page_number(
        self,
        page_count
    ):

        # ไม่แสดงเลขหน้า
        pass


# =========================================================
# PDF STYLES
# =========================================================

def create_pdf_styles():

    if not FONT_READY:

        raise RuntimeError(
            FONT_ERROR
            or "Font ไม่พร้อมใช้งาน"
        )

    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        fontName="THSarabunBold",
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        spaceAfter=2 * mm,
    )

    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        fontName="THSarabun",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=7 * mm,
    )

    styles["meta"] = ParagraphStyle(
        "meta",
        fontName="THSarabun",
        fontSize=15,
        leading=20,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )

    styles["section"] = ParagraphStyle(
        "section",
        fontName="THSarabunBold",
        fontSize=18,
        leading=23,
        alignment=TA_LEFT,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    styles["subsection"] = ParagraphStyle(
        "subsection",
        fontName="THSarabunBold",
        fontSize=16,
        leading=21,
        alignment=TA_LEFT,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )

    styles["body"] = ParagraphStyle(
        "body",
        fontName="THSarabun",
        fontSize=15,
        leading=23,
        alignment=TA_LEFT,
        firstLineIndent=10 * mm,
        spaceAfter=3.5 * mm,
    )

    styles["body_no_indent"] = ParagraphStyle(
        "body_no_indent",
        fontName="THSarabun",
        fontSize=15,
        leading=23,
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceAfter=3.5 * mm,
    )

    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="THSarabun",
        fontSize=15,
        leading=22,
        leftIndent=7 * mm,
        firstLineIndent=-4 * mm,
        spaceAfter=2.5 * mm,
    )

    styles["question"] = ParagraphStyle(
        "question",
        fontName="THSarabun",
        fontSize=16,
        leading=23,
        alignment=TA_LEFT,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=2 * mm,
    )

    styles["answer"] = ParagraphStyle(
        "answer",
        fontName="THSarabun",
        fontSize=15,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )

    styles["choice"] = ParagraphStyle(
        "choice",
        fontName="THSarabun",
        fontSize=15,
        leading=22,
        alignment=TA_LEFT,
        leftIndent=10 * mm,
        firstLineIndent=0,
        spaceAfter=1.5 * mm,
    )

    styles["small"] = ParagraphStyle(
        "small",
        fontName="THSarabun",
        fontSize=14,
        leading=19,
        alignment=TA_LEFT,
    )

    return styles


# =========================================================
# PDF CONTENT HELPERS
# =========================================================

def add_paragraphs(
    story,
    text,
    styles,
    indent=True
):

    text = clean_text(
        text
    )

    if not text:
        return

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    2 * mm
                )
            )

            continue

        line = strip_markdown(
            line
        )

        story.append(
            Paragraph(
                pdf_escape(line),
                styles[
                    "body"
                    if indent
                    else "body_no_indent"
                ]
            )
        )


def add_bullets(
    story,
    items,
    styles
):

    if not isinstance(
        items,
        list
    ):
        return

    for item in items:

        text = clean_text(
            item
        )

        if not text:
            continue

        story.append(
            Paragraph(
                "– " + pdf_escape(text),
                styles["bullet"]
            )
        )


def get_summary_value(
    summary,
    key,
    default=""
):

    value = summary.get(
        key,
        default
    )

    return clean_text(
        value
    )


# =========================================================
# PDF HEADER
# =========================================================

def build_document_header(
    story,
    data,
    styles,
    title
):

    summary = data.get(
        "summary",
        {}
    )

    teacher = clean_text(
        data.get(
            "teacher_name",
            ""
        )
    )

    subject = get_summary_value(
        summary,
        "subject"
    )

    grade = get_summary_value(
        summary,
        "grade"
    )

    topic = get_summary_value(
        summary,
        "topic"
    )

    duration = get_summary_value(
        summary,
        "duration"
    )

    story.append(
        Paragraph(
            pdf_escape(title),
            styles["title"]
        )
    )

    story.append(
        Paragraph(
            "เรื่อง " +
            pdf_escape(topic),
            styles["subtitle"]
        )
    )

    meta_parts = []

    if subject:
        meta_parts.append(
            "วิชา " +
            pdf_escape(subject)
        )

    if grade:
        meta_parts.append(
            "ระดับชั้น " +
            pdf_escape(grade)
        )

    if duration:
        meta_parts.append(
            "เวลา " +
            pdf_escape(duration)
        )

    if teacher:
        meta_parts.append(
            "ครูผู้สอน " +
            pdf_escape(teacher)
        )

    meta = "    |    ".join(
        meta_parts
    )

    story.append(
        Paragraph(
            meta,
            styles["meta"]
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm
        )
    )


# =========================================================
# LESSON PLAN PDF
# =========================================================

def build_lesson_pdf(
    story,
    data,
    styles
):

    lesson = data.get(
        "lesson_plan",
        {}
    )

    build_document_header(
        story,
        data,
        styles,
        "แผนการจัดการเรียนรู้"
    )

    # -----------------------------------------------------
    # 1 OBJECTIVE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "1. จุดประสงค์การเรียนรู้",
            styles["section"]
        )
    )

    add_bullets(
        story,
        lesson.get(
            "objective",
            []
        ),
        styles
    )

    # -----------------------------------------------------
    # 2 CONTENT
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "2. เนื้อหาที่ใช้สอน",
            styles["section"]
        )
    )

    add_paragraphs(
        story,
        lesson.get(
            "content",
            ""
        ),
        styles,
        indent=True
    )

    # -----------------------------------------------------
    # IMPORTANT POINTS
    # -----------------------------------------------------

    important = lesson.get(
        "important_points",
        []
    )

    if important:

        story.append(
            Paragraph(
                "สาระสำคัญ",
                styles["subsection"]
            )
        )

        add_bullets(
            story,
            important,
            styles
        )

    # -----------------------------------------------------
    # EXAMPLES
    # -----------------------------------------------------

    examples = lesson.get(
        "examples",
        []
    )

    if examples:

        story.append(
            Paragraph(
                "ตัวอย่างสำหรับใช้สอน",
                styles["subsection"]
            )
        )

        add_bullets(
            story,
            examples,
            styles
        )

    # -----------------------------------------------------
    # STEPS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "3. ขั้นตอนการจัดการเรียนรู้",
            styles["section"]
        )
    )

    steps = lesson.get(
        "steps",
        []
    )

    for index, step in enumerate(
        steps,
        start=1
    ):

        if not isinstance(
            step,
            dict
        ):
            continue

        time = clean_text(
            step.get(
                "time",
                ""
            )
        )

        title = clean_text(
            step.get(
                "title",
                ""
            )
        )

        detail = clean_text(
            step.get(
                "detail",
                ""
            )
        )

        heading = (
            f"{index}. "
            f"{time} "
            f"{title}"
        )

        story.append(
            Paragraph(
                pdf_escape(heading),
                styles["subsection"]
            )
        )

        add_paragraphs(
            story,
            detail,
            styles,
            indent=True
        )

    # -----------------------------------------------------
    # ASSESSMENT
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "4. การประเมินผล",
            styles["section"]
        )
    )

    add_paragraphs(
        story,
        lesson.get(
            "assessment",
            ""
        ),
        styles,
        indent=True
    )


# =========================================================
# WORKSHEET PDF
# =========================================================

def build_worksheet_pdf(
    story,
    data,
    styles
):

    worksheet = data.get(
        "worksheet",
        []
    )

    build_document_header(
        story,
        data,
        styles,
        "ใบงาน"
    )

    story.append(
        Paragraph(
            "ชื่อ-สกุล ................................................................................................",
            styles["body_no_indent"]
        )
    )

    story.append(
        Paragraph(
            "ชั้น .................... เลขที่ .................... วันที่ ....................",
            styles["body_no_indent"]
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    story.append(
        Paragraph(
            "คำชี้แจง",
            styles["subsection"]
        )
    )

    story.append(
        Paragraph(
            "ให้นักเรียนอ่านคำถามแต่ละข้อ และเขียนคำตอบลงในพื้นที่ที่กำหนด",
            styles["body"]
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm
        )
    )

    for index, item in enumerate(
        worksheet,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):
            continue

        no = safe_int(
            item.get(
                "no",
                index
            ),
            index
        )

        question = clean_text(
            item.get(
                "question",
                ""
            )
        )

        answer_lines = (
            "คำตอบ............................................................................................"
        )

        story.append(
            Paragraph(
                f"ข้อ {no}  " +
                pdf_escape(question),
                styles["question"]
            )
        )

        story.append(
            Paragraph(
                answer_lines,
                styles["answer"]
            )
        )

        # เส้นคำตอบเพิ่มเพียงบรรทัดเดียว
        story.append(
            Spacer(
                1,
                3 * mm
            )
        )


# =========================================================
# QUIZ PDF
# =========================================================

def normalize_choice(
    index
):

    choices = [
        "ก.",
        "ข.",
        "ค.",
        "ง.",
    ]

    if 0 <= index < len(
        choices
    ):
        return choices[index]

    return f"{index + 1}."


def build_quiz_pdf(
    story,
    data,
    styles
):

    quiz = data.get(
        "quiz",
        []
    )

    build_document_header(
        story,
        data,
        styles,
        "แบบทดสอบ"
    )

    story.append(
        Paragraph(
            "ชื่อ-สกุล ................................................................................................",
            styles["body_no_indent"]
        )
    )

    story.append(
        Paragraph(
            "ชั้น .................... เลขที่ .................... วันที่ ....................",
            styles["body_no_indent"]
        )

    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    story.append(
        Paragraph(
            "คำชี้แจง",
            styles["subsection"]
        )
    )

    story.append(
        Paragraph(
            "ให้นักเรียนทำแบบทดสอบทุกข้อ และเลือกหรือเขียนคำตอบให้ถูกต้อง",
            styles["body"]
        )
    )

    for index, item in enumerate(
        quiz,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):
            continue

        no = safe_int(
            item.get(
                "no",
                index
            ),
            index
        )

        question = clean_text(
            item.get(
                "question",
                ""
            )
        )

        qtype = clean_text(
            item.get(
                "type",
                ""
            )
        )

        block = []

        block.append(
            Paragraph(
                f"ข้อ {no}  " +
                pdf_escape(question),
                styles["question"]
            )
        )

        options = item.get(
            "options",
            []
        )

        if (
            qtype == "multiple_choice"
            and isinstance(
                options,
                list
            )
        ):

            for option_index, option in enumerate(
                options[:4]
            ):

                label = normalize_choice(
                    option_index
                )

                block.append(
                    Paragraph(
                        label +
                        " " +
                        pdf_escape(
                            clean_text(
                                option
                            )
                        ),
                        styles["choice"]
                    )
                )

        else:

            block.append(
                Paragraph(
                    "คำตอบ............................................................................................",
                    styles["answer"]
                )
            )

        block.append(
            Spacer(
                1,
                5 * mm
            )
        )

        story.append(
            KeepTogether(
                block
            )
        )


# =========================================================
# ANSWER PDF
# =========================================================

def build_answer_pdf(
    story,
    data,
    styles
):

    worksheet = data.get(
        "worksheet",
        []
    )

    quiz = data.get(
        "quiz",
        []
    )

    build_document_header(
        story,
        data,
        styles,
        "เฉลย"
    )

    if worksheet:

        story.append(
            Paragraph(
                "เฉลยใบงาน",
                styles["section"]
            )
        )

        for index, item in enumerate(
            worksheet,
            start=1
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            no = safe_int(
                item.get(
                    "no",
                    index
                ),
                index
            )

            answer = clean_text(
                item.get(
                    "answer",
                    ""
                )
            )

            story.append(
                Paragraph(
                    f"ข้อ {no}",
                    styles["subsection"]
                )
            )

            add_paragraphs(
                story,
                answer,
                styles,
                indent=True
            )

    if quiz:

        story.append(
            Paragraph(
                "เฉลยแบบทดสอบ",
                styles["section"]
            )
        )

        for index, item in enumerate(
            quiz,
            start=1
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            no = safe_int(
                item.get(
                    "no",
                    index
                ),
                index
            )

            answer = clean_text(
                item.get(
                    "answer",
                    ""
                )
            )

            explanation = clean_text(
                item.get(
                    "explanation",
                    ""
                )
            )

            story.append(
                Paragraph(
                    f"ข้อ {no} — " +
                    pdf_escape(answer),
                    styles["subsection"]
                )
            )

            if explanation:

                add_paragraphs(
                    story,
                    explanation,
                    styles,
                    indent=True
                )


# =========================================================
# PDF GENERATOR
# =========================================================

def create_pdf(
    data: dict
):

    if not FONT_READY:

        raise RuntimeError(
            FONT_ERROR
            or "ไม่พบ Font"
        )

    styles = create_pdf_styles()

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title="เอกสารการเรียนการสอน",

        author=clean_text(
            data.get(
                "teacher_name",
                ""
            )
        ),
    )

    story = []

    # -----------------------------------------------------
    # LESSON
    # -----------------------------------------------------

    build_lesson_pdf(
        story,
        data,
        styles
    )

    story.append(
        PageBreak()
    )

    # -----------------------------------------------------
    # WORKSHEET
    # -----------------------------------------------------

    build_worksheet_pdf(
        story,
        data,
        styles
    )

    story.append(
        PageBreak()
    )

    # -----------------------------------------------------
    # QUIZ
    # -----------------------------------------------------

    build_quiz_pdf(
        story,
        data,
        styles
    )

    story.append(
        PageBreak()
    )

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    build_answer_pdf(
        story,
        data,
        styles
    )

    document.build(
        story,
        canvasmaker=NumberedCanvas
    )

    buffer.seek(0)

    return buffer.read()


# =========================================================
# PDF API
# =========================================================

@app.post("/api/pdf")
def make_pdf(request: PDFRequest):

    try:

        data = ensure_structure(
            request.data
        )

        pdf = create_pdf(
            data
        )

        return Response(

            content=pdf,

            media_type="application/pdf",

            headers={
                "Content-Disposition":
                    'attachment; filename="teacher-pack.pdf"'
            }
        )

    except Exception as e:

        print(
            traceback.format_exc()
        )

        raise HTTPException(

            status_code=500,

            detail=f"สร้าง PDF ไม่สำเร็จ: {str(e)}"
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
