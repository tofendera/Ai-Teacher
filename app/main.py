import os
import json
import html
import re
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openai import OpenAI

from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================
# PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

FONT_REGULAR = APP_DIR / "THSarabun.ttf"
FONT_BOLD = APP_DIR / "THSarabunBold.ttf"


# ============================================================
# FONT
# ============================================================

if not FONT_REGULAR.exists():
    raise RuntimeError(
        f"ไม่พบไฟล์ Font: {FONT_REGULAR}"
    )

if not FONT_BOLD.exists():
    raise RuntimeError(
        f"ไม่พบไฟล์ Font: {FONT_BOLD}"
    )

pdfmetrics.registerFont(
    TTFont("THSarabun", str(FONT_REGULAR))
)

pdfmetrics.registerFont(
    TTFont("THSarabunBold", str(FONT_BOLD))
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.5.0",
)


# ============================================================
# STATIC
# ============================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# ============================================================
# OPENAI
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None


# ============================================================
# REQUEST
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=2,
        max_length=500
    )

    teacher_name: str = Field(
        default="",
        max_length=100
    )

    question_count: int = Field(
        default=10,
        ge=5,
        le=30
    )

    question_types: list[str] = Field(
        default_factory=lambda: [
            "multiple_choice"
        ]
    )

    difficulty: str = "mixed"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
คุณคือผู้ช่วยจัดทำเอกสารการเรียนการสอนสำหรับครูไทย

หน้าที่คือสร้างชุดเอกสารการสอนจากหัวข้อที่ครูกำหนด

ต้องสร้างข้อมูลดังนี้

1. ข้อมูลสรุป
2. จุดประสงค์การเรียนรู้
3. เนื้อหาที่ใช้สอน
4. ตัวอย่างสำหรับใช้สอน
5. คำถามชวนคิด
6. ขั้นตอนการจัดการเรียนรู้
7. ใบงาน
8. แบบทดสอบ
9. เฉลย

ข้อกำหนดสำคัญ

- ใช้ภาษาไทยเป็นหลัก
- เนื้อหาต้องเหมาะกับระดับชั้น
- เนื้อหาต้องถูกต้อง
- ตรวจสอบคำตอบก่อนส่ง
- ใช้เลขอารบิก 1, 2, 3, 4
- ห้ามใช้เลขไทย
- ห้ามใส่ Emoji ในข้อมูลที่จะนำไปสร้าง PDF
- ห้ามใส่สัญลักษณ์ตกแต่งที่ไม่จำเป็น
- ห้ามใส่คำว่า AI ครูผู้ช่วย
- ห้ามใส่คำว่า AI-Teacher ในเนื้อหา
- ห้ามใส่ชื่อครูเอง ถ้าได้รับชื่อครูจากผู้ใช้ ให้ระบบนำชื่อไปใส่เอง
- ห้ามสร้างตารางสำหรับเนื้อหา
- เขียนเนื้อหาเป็นย่อหน้า
- หัวข้อควรสั้นและชัดเจน

รูปแบบใบงาน

แต่ละข้อให้ส่ง question แยกออกมา
ห้ามใส่เลขข้อซ้ำใน question

ตัวอย่าง

question:
"เขียนชื่อทักษะพื้นฐานของฟุตบอล 4 อย่างที่เรียนวันนี้"

answer:
"ตัวอย่างคำตอบ ..."

สำหรับปรนัย

ต้องมีตัวเลือก 4 ตัวเลือก
และมีคำตอบที่ถูกต้องเพียง 1 ตัวเลือก

ตัวเลือกต้องอยู่ใน options
โดยไม่ต้องใส่ ก. ข. ค. ง. ในข้อความตัวเลือก

สำหรับคำตอบ

answer ต้องเป็นคำตอบที่ถูกต้อง
explanation ต้องเป็นคำอธิบายสั้น ๆ ที่เข้าใจง่าย

ขั้นตอนการจัดการเรียนรู้

แต่ละขั้นต้องมี
- time
- title
- detail

เขียน detail เป็นข้อความธรรมชาติ ไม่ใช่ข้อมูลยาวติดกัน

ให้สร้างเอกสารที่สามารถนำไปจัดหน้าเป็นเอกสารราชการ/เอกสารโรงเรียนได้จริง
"""


# ============================================================
# JSON SCHEMA
# ============================================================

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {

        "summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grade": {"type": "string"},
                "subject": {"type": "string"},
                "topic": {"type": "string"},
                "duration": {"type": "string"},
            },
            "required": [
                "grade",
                "subject",
                "topic",
                "duration",
            ],
        },

        "lesson_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {

                "objective": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },

                "content": {
                    "type": "string"
                },

                "examples": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },

                "thinking_questions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },

                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "time": {
                                "type": "string"
                            },
                            "title": {
                                "type": "string"
                            },
                            "detail": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "time",
                            "title",
                            "detail",
                        ],
                    },
                },

                "assessment": {
                    "type": "string"
                },

                "teacher_notes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                },
            },

            "required": [
                "objective",
                "content",
                "examples",
                "thinking_questions",
                "steps",
                "assessment",
                "teacher_notes",
            ],
        },

        "worksheet": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "no": {
                        "type": "integer"
                    },
                    "question": {
                        "type": "string"
                    },
                    "answer": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": [
                    "no",
                    "question",
                    "answer",
                    "type",
                    "options",
                ],
            },
        },

        "quiz": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "no": {
                        "type": "integer"
                    },
                    "question": {
                        "type": "string"
                    },
                    "answer": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": [
                    "no",
                    "question",
                    "answer",
                    "explanation",
                    "type",
                    "options",
                ],
            },
        },
    },

    "required": [
        "summary",
        "lesson_plan",
        "worksheet",
        "quiz",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value
    )

    return value.strip()


def clean_data(data):

    if not isinstance(data, dict):
        return data

    summary = data.get("summary", {})

    for key in [
        "grade",
        "subject",
        "topic",
        "duration",
    ]:
        summary[key] = clean_text(
            summary.get(key, "")
        )

    data["summary"] = summary

    lesson = data.get(
        "lesson_plan",
        {}
    )

    lesson["content"] = clean_text(
        lesson.get("content", "")
    )

    lesson["assessment"] = clean_text(
        lesson.get("assessment", "")
    )

    lesson["objective"] = [
        clean_text(x)
        for x in lesson.get(
            "objective",
            []
        )
        if clean_text(x)
    ]

    lesson["examples"] = [
        clean_text(x)
        for x in lesson.get(
            "examples",
            []
        )
        if clean_text(x)
    ]

    lesson["thinking_questions"] = [
        clean_text(x)
        for x in lesson.get(
            "thinking_questions",
            []
        )
        if clean_text(x)
    ]

    lesson["teacher_notes"] = [
        clean_text(x)
        for x in lesson.get(
            "teacher_notes",
            []
        )
        if clean_text(x)
    ]

    for step in lesson.get(
        "steps",
        []
    ):
        step["time"] = clean_text(
            step.get("time", "")
        )
        step["title"] = clean_text(
            step.get("title", "")
        )
        step["detail"] = clean_text(
            step.get("detail", "")
        )

    data["lesson_plan"] = lesson

    for item in data.get(
        "worksheet",
        []
    ):
        item["question"] = clean_text(
            item.get("question", "")
        )
        item["answer"] = clean_text(
            item.get("answer", "")
        )

        item["options"] = [
            clean_text(x)
            for x in item.get(
                "options",
                []
            )
            if clean_text(x)
        ]

    for item in data.get(
        "quiz",
        []
    ):
        item["question"] = clean_text(
            item.get("question", "")
        )
        item["answer"] = clean_text(
            item.get("answer", "")
        )
        item["explanation"] = clean_text(
            item.get("explanation", "")
        )

        item["options"] = [
            clean_text(x)
            for x in item.get(
                "options",
                []
            )
            if clean_text(x)
        ]

    return data


def esc(value):
    return html.escape(
        clean_text(value)
    )


def paragraph_text(value):
    """
    แปลงข้อความให้ ReportLab
    รักษาการขึ้นบรรทัดและย่อหน้า
    """

    value = clean_text(value)

    parts = re.split(
        r"\n\s*\n",
        value
    )

    result = []

    for part in parts:

        part = re.sub(
            r"[ \t]+",
            " ",
            part
        )

        part = part.strip()

        if not part:
            continue

        lines = part.split("\n")

        line_text = "<br/>".join(
            esc(line)
            for line in lines
        )

        result.append(
            line_text
        )

    return "<br/><br/>".join(
        result
    )


def normalize_question_number(
    question,
    number
):
    """
    ป้องกัน AI ใส่เลขข้อซ้ำ
    """

    question = clean_text(
        question
    )

    question = re.sub(
        r"^\s*(ข้อ\s*)?\d+[\.\):\-]?\s*",
        "",
        question,
        flags=re.IGNORECASE
    )

    return f"ข้อ {number}  {question}"


# ============================================================
# GENERATE AI
# ============================================================

@app.post("/api/generate")
def generate_pack(req: GenerateRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "ยังไม่ได้ตั้งค่า OPENAI_API_KEY "
                "ใน Render Environment Variables"
            )
        )

    teacher_name = clean_text(
        req.teacher_name
    )

    if not teacher_name:
        raise HTTPException(
            status_code=400,
            detail="กรุณากรอกชื่อครูผู้สอน"
        )

    user_prompt = f"""
หัวข้อที่ครูต้องการสอน:
{req.prompt}

ชื่อครูผู้สอน:
{teacher_name}

จำนวนข้อ:
{req.question_count}

รูปแบบข้อสอบที่เลือก:
{", ".join(req.question_types)}

ระดับความยาก:
{req.difficulty}

สร้างชุดเอกสารการเรียนการสอนให้ครบถ้วน

ต้องสร้างใบงานจำนวน {req.question_count} ข้อ
และแบบทดสอบจำนวน {req.question_count} ข้อ

หากเลือกรูปแบบปรนัย ต้องมีตัวเลือก 4 ตัวเลือก
หากมีหลายรูปแบบ ให้ผสมตามรูปแบบที่เลือก

จัดเนื้อหาให้เหมาะกับระดับชั้น
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,

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

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "teacher_pack",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "AI ไม่ส่งข้อมูลกลับมา"
            )

        data = json.loads(
            content
        )

        data = clean_data(
            data
        )

        # บังคับข้อมูลจากผู้ใช้
        # ไม่ให้ AI เปลี่ยนชื่อครู
        data["teacher_name"] = (
            teacher_name
        )

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"สร้างชุดการสอนไม่สำเร็จ: {str(e)}"
        )


# ============================================================
# PDF STYLES
# ============================================================

STYLE_TITLE = ParagraphStyle(
    "Title",
    fontName="THSarabunBold",
    fontSize=24,
    leading=28,
    alignment=TA_CENTER,
    spaceAfter=4 * mm,
)

STYLE_SUBTITLE = ParagraphStyle(
    "Subtitle",
    fontName="THSarabun",
    fontSize=17,
    leading=21,
    alignment=TA_CENTER,
    spaceAfter=7 * mm,
)

STYLE_META = ParagraphStyle(
    "Meta",
    fontName="THSarabun",
    fontSize=14,
    leading=18,
    alignment=TA_LEFT,
    spaceAfter=8 * mm,
)

STYLE_HEADING = ParagraphStyle(
    "Heading",
    fontName="THSarabunBold",
    fontSize=18,
    leading=23,
    spaceBefore=7 * mm,
    spaceAfter=3 * mm,
)

STYLE_SUBHEADING = ParagraphStyle(
    "SubHeading",
    fontName="THSarabunBold",
    fontSize=16,
    leading=21,
    spaceBefore=5 * mm,
    spaceAfter=2 * mm,
)

STYLE_BODY = ParagraphStyle(
    "Body",
    fontName="THSarabun",
    fontSize=15,
    leading=21,
    alignment=TA_LEFT,
    firstLineIndent=10 * mm,
    spaceAfter=4 * mm,
)

STYLE_BODY_NO_INDENT = ParagraphStyle(
    "BodyNoIndent",
    fontName="THSarabun",
    fontSize=15,
    leading=21,
    alignment=TA_LEFT,
    spaceAfter=3 * mm,
)

STYLE_BULLET = ParagraphStyle(
    "Bullet",
    fontName="THSarabun",
    fontSize=15,
    leading=21,
    leftIndent=8 * mm,
    firstLineIndent=-5 * mm,
    spaceAfter=2.5 * mm,
)

STYLE_QUESTION = ParagraphStyle(
    "Question",
    fontName="THSarabun",
    fontSize=16,
    leading=23,
    alignment=TA_LEFT,
    spaceBefore=4 * mm,
    spaceAfter=2 * mm,
)

STYLE_OPTION = ParagraphStyle(
    "Option",
    fontName="THSarabun",
    fontSize=15,
    leading=21,
    leftIndent=15 * mm,
    firstLineIndent=-8 * mm,
    spaceAfter=1.5 * mm,
)

STYLE_ANSWER_LINE = ParagraphStyle(
    "AnswerLine",
    fontName="THSarabun",
    fontSize=15,
    leading=21,
    spaceBefore=2 * mm,
    spaceAfter=5 * mm,
)

STYLE_SMALL = ParagraphStyle(
    "Small",
    fontName="THSarabun",
    fontSize=14,
    leading=19,
    spaceAfter=2 * mm,
)


# ============================================================
# PDF PAGE
# ============================================================

def draw_page(canvas, doc):
    """
    ตั้งค่าหน้ากระดาษ
    ไม่ใส่เลขหน้า
    """

    canvas.saveState()

    # ไม่ใส่เลขหน้า
    # ไม่ใส่ footer

    canvas.restoreState()


# ============================================================
# PDF HELPERS
# ============================================================

def add_heading(
    story,
    text
):
    story.append(
        Paragraph(
            esc(text),
            STYLE_HEADING
        )
    )


def add_subheading(
    story,
    text
):
    story.append(
        Paragraph(
            esc(text),
            STYLE_SUBHEADING
        )
    )


def add_body(
    story,
    text,
    indent=True
):

    style = (
        STYLE_BODY
        if indent
        else STYLE_BODY_NO_INDENT
    )

    story.append(
        Paragraph(
            paragraph_text(text),
            style
        )
    )


def add_bullet(
    story,
    text
):

    story.append(
        Paragraph(
            f"- {paragraph_text(text)}",
            STYLE_BULLET
        )
    )


# ============================================================
# PDF LESSON
# ============================================================

def build_lesson_pdf(
    story,
    data
):

    summary = data.get(
        "summary",
        {}
    )

    lesson = data.get(
        "lesson_plan",
        {}
    )

    teacher_name = data.get(
        "teacher_name",
        ""
    )

    # ----------------------------
    # Title
    # ----------------------------

    story.append(
        Paragraph(
            "แผนการจัดการเรียนรู้",
            STYLE_TITLE
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {esc(summary.get('topic', ''))}",
            STYLE_SUBTITLE
        )
    )

    # ----------------------------
    # Metadata
    # ----------------------------

    meta = (
        f"วิชา {esc(summary.get('subject', ''))}"
        f"　|　"
        f"ระดับชั้น {esc(summary.get('grade', ''))}"
        f"　|　"
        f"เวลา {esc(summary.get('duration', ''))}"
        f"　|　"
        f"ครูผู้สอน {esc(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta,
            STYLE_META
        )
    )

    # ----------------------------
    # 1 Objective
    # ----------------------------

    add_heading(
        story,
        "1. จุดประสงค์การเรียนรู้"
    )

    for item in lesson.get(
        "objective",
        []
    ):
        add_bullet(
            story,
            item
        )

    # ----------------------------
    # 2 Content
    # ----------------------------

    add_heading(
        story,
        "2. เนื้อหาที่ใช้สอน"
    )

    add_body(
        story,
        lesson.get(
            "content",
            ""
        ),
        indent=True
    )

    # ----------------------------
    # Important
    # ----------------------------

    if lesson.get(
        "examples"
    ):

        add_subheading(
            story,
            "สาระสำคัญ"
        )

        for item in lesson.get(
            "examples",
            []
        ):
            add_body(
                story,
                item,
                indent=True
            )

    # ----------------------------
    # Examples
    # ----------------------------

    if lesson.get(
        "thinking_questions"
    ):

        add_subheading(
            story,
            "คำถามชวนคิด"
        )

        for item in lesson.get(
            "thinking_questions",
            []
        ):
            add_bullet(
                story,
                item
            )

    # ----------------------------
    # 3 Teaching Steps
    # ----------------------------

    add_heading(
        story,
        "3. ขั้นตอนการจัดการเรียนรู้"
    )

    for index, step in enumerate(
        lesson.get("steps", []),
        start=1
    ):

        title = (
            f"{index}. "
            f"{step.get('title', '')}"
        )

        text = (
            f"{step.get('time', '')} "
            f"{step.get('detail', '')}"
        )

        story.append(
            Paragraph(
                esc(title),
                STYLE_SUBHEADING
            )
        )

        add_body(
            story,
            text,
            indent=True
        )

    # ----------------------------
    # Assessment
    # ----------------------------

    add_heading(
        story,
        "4. การประเมินผล"
    )

    add_body(
        story,
        lesson.get(
            "assessment",
            ""
        ),
        indent=True
    )

    # ----------------------------
    # Teacher notes
    # ----------------------------

    if lesson.get(
        "teacher_notes"
    ):

        add_heading(
            story,
            "5. ข้อเสนอแนะสำหรับครู"
        )

        for item in lesson.get(
            "teacher_notes",
            []
        ):
            add_bullet(
                story,
                item
            )


# ============================================================
# PDF WORKSHEET
# ============================================================

def build_worksheet_pdf(
    story,
    data
):

    summary = data.get(
        "summary",
        {}
    )

    teacher_name = data.get(
        "teacher_name",
        ""
    )

    story.append(
        Paragraph(
            "ใบงาน",
            STYLE_TITLE
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {esc(summary.get('topic', ''))}",
            STYLE_SUBTITLE
        )
    )

    meta = (
        f"วิชา {esc(summary.get('subject', ''))}"
        f"　|　"
        f"ระดับชั้น {esc(summary.get('grade', ''))}"
        f"　|　"
        f"เวลา {esc(summary.get('duration', ''))}"
        f"　|　"
        f"ครูผู้สอน {esc(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta,
            STYLE_META
        )
    )

    # ----------------------------
    # Student info
    # ----------------------------

    add_body(
        story,
        "ชื่อ-สกุล ................................................................................................................",
        indent=False
    )

    add_body(
        story,
        "ชั้น ........................................ เลขที่ ................ วันที่ ........................................",
        indent=False
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    add_subheading(
        story,
        "คำชี้แจง"
    )

    add_body(
        story,
        "ให้นักเรียนอ่านคำถามแต่ละข้อ และเขียนคำตอบลงในพื้นที่ที่กำหนด",
        indent=False
    )

    # ----------------------------
    # Questions
    # ----------------------------

    for item in data.get(
        "worksheet",
        []
    ):

        no = item.get(
            "no",
            0
        )

        question = normalize_question_number(
            item.get(
                "question",
                ""
            ),
            no
        )

        options = item.get(
            "options",
            []
        )

        block = []

        block.append(
            Paragraph(
                paragraph_text(question),
                STYLE_QUESTION
            )
        )

        if options:

            letters = [
                "ก.",
                "ข.",
                "ค.",
                "ง.",
            ]

            for i, option in enumerate(
                options[:4]
            ):

                letter = (
                    letters[i]
                    if i < len(letters)
                    else ""
                )

                block.append(
                    Paragraph(
                        f"{letter} "
                        f"{paragraph_text(option)}",
                        STYLE_OPTION
                    )
                )

        # คำตอบเพียงบรรทัดเดียว
        block.append(
            Paragraph(
                "คำตอบ........................................................................................................................",
                STYLE_ANSWER_LINE
            )
        )

        story.append(
            KeepTogether(
                block
            )
        )


# ============================================================
# PDF QUIZ
# ============================================================

def build_quiz_pdf(
    story,
    data
):

    summary = data.get(
        "summary",
        {}
    )

    teacher_name = data.get(
        "teacher_name",
        ""
    )

    story.append(
        Paragraph(
            "แบบทดสอบ",
            STYLE_TITLE
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {esc(summary.get('topic', ''))}",
            STYLE_SUBTITLE
        )
    )

    meta = (
        f"วิชา {esc(summary.get('subject', ''))}"
        f"　|　"
        f"ระดับชั้น {esc(summary.get('grade', ''))}"
        f"　|　"
        f"เวลา {esc(summary.get('duration', ''))}"
        f"　|　"
        f"ครูผู้สอน {esc(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta,
            STYLE_META
        )
    )

    add_body(
        story,
        "ชื่อ-สกุล ................................................................................................................",
        indent=False
    )

    add_body(
        story,
        "ชั้น ........................................ เลขที่ ................ วันที่ ........................................",
        indent=False
    )

    add_subheading(
        story,
        "คำชี้แจง"
    )

    add_body(
        story,
        "ให้นักเรียนทำแบบทดสอบทุกข้อ และเลือกหรือเขียนคำตอบให้ถูกต้อง",
        indent=False
    )

    for item in data.get(
        "quiz",
        []
    ):

        no = item.get(
            "no",
            0
        )

        question = normalize_question_number(
            item.get(
                "question",
                ""
            ),
            no
        )

        block = [
            Paragraph(
                paragraph_text(question),
                STYLE_QUESTION
            )
        ]

        options = item.get(
            "options",
            []
        )

        if options:

            letters = [
                "ก.",
                "ข.",
                "ค.",
                "ง.",
            ]

            for i, option in enumerate(
                options[:4]
            ):

                letter = (
                    letters[i]
                    if i < len(letters)
                    else ""
                )

                block.append(
                    Paragraph(
                        f"{letter} "
                        f"{paragraph_text(option)}",
                        STYLE_OPTION
                    )
                )

        block.append(
            Paragraph(
                "คำตอบ........................................................................................................................",
                STYLE_ANSWER_LINE
            )
        )

        story.append(
            KeepTogether(
                block
            )
        )


# ============================================================
# PDF ANSWER
# ============================================================

def build_answer_pdf(
    story,
    data
):

    summary = data.get(
        "summary",
        {}
    )

    teacher_name = data.get(
        "teacher_name",
        ""
    )

    story.append(
        Paragraph(
            "เฉลย",
            STYLE_TITLE
        )
    )

    story.append(
        Paragraph(
            f"เรื่อง {esc(summary.get('topic', ''))}",
            STYLE_SUBTITLE
        )
    )

    meta = (
        f"วิชา {esc(summary.get('subject', ''))}"
        f"　|　"
        f"ระดับชั้น {esc(summary.get('grade', ''))}"
        f"　|　"
        f"เวลา {esc(summary.get('duration', ''))}"
        f"　|　"
        f"ครูผู้สอน {esc(teacher_name)}"
    )

    story.append(
        Paragraph(
            meta,
            STYLE_META
        )
    )

    add_heading(
        story,
        "เฉลยใบงาน"
    )

    for item in data.get(
        "worksheet",
        []
    ):

        no = item.get(
            "no",
            0
        )

        answer = item.get(
            "answer",
            ""
        )

        story.append(
            Paragraph(
                f"ข้อ {no}",
                STYLE_SUBHEADING
            )
        )

        add_body(
            story,
            answer,
            indent=True
        )

    add_heading(
        story,
        "เฉลยแบบทดสอบ"
    )

    for item in data.get(
        "quiz",
        []
    ):

        no = item.get(
            "no",
            0
        )

        answer = item.get(
            "answer",
            ""
        )

        explanation = item.get(
            "explanation",
            ""
        )

        story.append(
            Paragraph(
                f"ข้อ {no}",
                STYLE_SUBHEADING
            )
        )

        add_body(
            story,
            f"คำตอบ: {answer}",
            indent=False
        )

        if explanation:
            add_body(
                story,
                explanation,
                indent=True
            )


# ============================================================
# CREATE PDF
# ============================================================

def create_pdf(data):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,

        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,

        title="เอกสารการเรียนการสอน",
        author=data.get(
            "teacher_name",
            ""
        ),
    )

    story = []

    # ----------------------------
    # Lesson
    # ----------------------------

    build_lesson_pdf(
        story,
        data
    )

    # ----------------------------
    # Worksheet
    # ----------------------------

    story.append(
        PageBreak()
    )

    build_worksheet_pdf(
        story,
        data
    )

    # ----------------------------
    # Quiz
    # ----------------------------

    story.append(
        PageBreak()
    )

    build_quiz_pdf(
        story,
        data
    )

    # ----------------------------
    # Answer
    # ----------------------------

    story.append(
        PageBreak()
    )

    build_answer_pdf(
        story,
        data
    )

    doc.build(
        story,
        onFirstPage=draw_page,
        onLaterPages=draw_page,
    )

    buffer.seek(0)

    return buffer


# ============================================================
# PDF ENDPOINT
# ============================================================

@app.post("/api/pdf")
def generate_pdf(data: dict):

    try:

        data = clean_data(
            data
        )

        if not data.get(
            "teacher_name"
        ):
            raise HTTPException(
                status_code=400,
                detail="ไม่พบชื่อครูผู้สอน"
            )

        pdf = create_pdf(
            data
        )

        filename = (
            "ชุดการสอน.pdf"
        )

        return StreamingResponse(
            pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{filename}"'
            },
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"สร้าง PDF ไม่สำเร็จ: {str(e)}"
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "1.5.0",
        "font_regular": FONT_REGULAR.exists(),
        "font_bold": FONT_BOLD.exists(),
        "static": STATIC_DIR.exists(),
        "openai_key": bool(
            OPENAI_API_KEY
        ),
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="ไม่พบ static/index.html"
        )

    return FileResponse(
        INDEX_FILE
    )
