import os
import json
import html
import re
import uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openai import OpenAI

from reportlab.lib import colors
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

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = APP_DIR / "THSarabun.ttf"
FONT_BOLDITALIC = APP_DIR / "THSarabun BoldItalic.ttf"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Teacher",
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
# FONT
# ============================================================

FONT_NAME = "THSarabun"

try:
    if FONT_REGULAR.exists():
        pdfmetrics.registerFont(
            TTFont(FONT_NAME, str(FONT_REGULAR))
        )
    else:
        FONT_NAME = "Helvetica"
except Exception:
    FONT_NAME = "Helvetica"


# ------------------------------------------------------------
# BoldItalic
#
# ลงทะเบียนไว้เพื่อรองรับไฟล์ที่ผู้ใช้เพิ่มมา
# แต่ไม่ใช้เป็นฟอนต์หลักสำหรับภาษาไทย
# เพราะบางไฟล์ BoldItalic ไม่มี glyph ภาษาไทยครบ
# ------------------------------------------------------------

FONT_BOLD_NAME = "THSarabunBoldItalic"

try:
    if FONT_BOLDITALIC.exists():
        pdfmetrics.registerFont(
            TTFont(
                FONT_BOLD_NAME,
                str(FONT_BOLDITALIC)
            )
        )
except Exception:
    FONT_BOLD_NAME = FONT_NAME


# ============================================================
# OPENAI
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=API_KEY
) if API_KEY else None

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
)


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

    difficulty: str = "mixed"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
คุณคือผู้ช่วยจัดทำเอกสารการเรียนการสอนสำหรับครูไทย

สร้างชุดการสอนให้ครบถ้วนตามหัวข้อที่ผู้ใช้ระบุ

ประกอบด้วย:

1. แผนการจัดการเรียนรู้
2. เนื้อหาที่ใช้สอน
3. ตัวอย่างสำหรับใช้สอน
4. คำถามชวนคิด
5. ใบงาน
6. แบบทดสอบ
7. เฉลย

ข้อกำหนดสำคัญ:

- ใช้ภาษาไทยเป็นหลัก
- เนื้อหาต้องเหมาะสมกับระดับชั้น
- เนื้อหาต้องถูกต้อง
- ตรวจสอบตัวเลขและคำตอบก่อนส่ง
- ใช้ตัวเลขอารบิก
- ห้ามใส่ Emoji ในข้อมูลที่จะนำไปสร้าง PDF
- ห้ามใช้สัญลักษณ์ตกแต่งที่ไม่จำเป็น
- เลขข้อให้เริ่มจาก 1
- ห้ามใส่เลขข้อซ้ำในข้อความคำถาม
- คำถามแบบปรนัยต้องมี 4 ตัวเลือก
- ตัวเลือกใช้ ก. ข. ค. ง.
- ต้องมีคำตอบที่ถูกต้องเพียง 1 ตัวเลือก
- ต้องมีคำอธิบายเฉลย
- ห้ามสร้างข้อมูลนอก JSON Schema
- ต้องตอบเป็น JSON เท่านั้น

รูปแบบคำถามในข้อมูล:

ห้ามเขียน:
"1. 1/2 + 1/4 มีค่าเท่าไร"

ให้เขียน:
"1/2 + 1/4 มีค่าเท่าไร"

เพราะระบบจะแสดงเลขข้อให้อัตโนมัติ

ห้ามใส่คำว่า "(ปรนัย)" ต่อท้ายคำถาม
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
                "grade": {
                    "type": "string"
                },
                "subject": {
                    "type": "string"
                },
                "topic": {
                    "type": "string"
                },
                "duration": {
                    "type": "string"
                },
                "teacher_name": {
                    "type": "string"
                }
            },
            "required": [
                "grade",
                "subject",
                "topic",
                "duration",
                "teacher_name"
            ]
        },

        "lesson_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {

                "objective": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
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
                            }
                        },
                        "required": [
                            "time",
                            "title",
                            "detail"
                        ]
                    }
                },

                "assessment": {
                    "type": "string"
                }
            },

            "required": [
                "objective",
                "steps",
                "assessment"
            ]
        },

        "teaching_content": {
            "type": "object",
            "additionalProperties": False,
            "properties": {

                "intro": {
                    "type": "string"
                },

                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {
                                "type": "string"
                            },
                            "explanation": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "title",
                            "explanation"
                        ]
                    }
                },

                "teacher_tips": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "thinking_questions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },

            "required": [
                "intro",
                "concepts",
                "examples",
                "teacher_tips",
                "thinking_questions"
            ]
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
                    }
                },
                "required": [
                    "no",
                    "question",
                    "answer"
                ]
            }
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

                    "type": {
                        "type": "string"
                    },

                    "question": {
                        "type": "string"
                    },

                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "answer": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    }
                },

                "required": [
                    "no",
                    "type",
                    "question",
                    "options",
                    "answer",
                    "explanation"
                ]
            }
        }
    },

    "required": [
        "summary",
        "lesson_plan",
        "teaching_content",
        "worksheet",
        "quiz"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    """
    ทำความสะอาดข้อความก่อนนำไปสร้าง PDF
    """

    if value is None:
        return ""

    value = str(value)

    value = value.replace(
        "\r\n",
        "\n"
    )

    value = value.replace(
        "\r",
        "\n"
    )

    # ลบ emoji / symbol บางประเภท
    value = re.sub(
        r"[\U00010000-\U0010ffff]",
        "",
        value
    )

    # ลบคำว่า (ปรนัย)
    value = value.replace(
        "(ปรนัย)",
        ""
    )

    return value.strip()


def safe_filename(name):
    name = re.sub(
        r"[^a-zA-Z0-9_\-ก-๙]+",
        "_",
        name
    )

    return name.strip("_") or "data"


def para_text(value):
    """
    Escape HTML สำหรับ ReportLab Paragraph
    """

    value = clean_text(value)

    value = html.escape(
        value,
        quote=False
    )

    value = value.replace(
        "\n",
        "<br/>"
    )

    return value


# ============================================================
# PDF STYLES
# ============================================================

def get_styles():

    return {

        "title": ParagraphStyle(
            "title",
            fontName=FONT_NAME,
            fontSize=25,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),

        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=FONT_NAME,
            fontSize=15,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),

        "heading": ParagraphStyle(
            "heading",
            fontName=FONT_NAME,
            fontSize=20,
            leading=25,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
        ),

        "subheading": ParagraphStyle(
            "subheading",
            fontName=FONT_NAME,
            fontSize=17,
            leading=22,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),

        "body": ParagraphStyle(
            "body",
            fontName=FONT_NAME,
            fontSize=15,
            leading=22,
            spaceAfter=3 * mm,
        ),

        "small": ParagraphStyle(
            "small",
            fontName=FONT_NAME,
            fontSize=13,
            leading=18,
            spaceAfter=2 * mm,
        ),

        "question": ParagraphStyle(
            "question",
            fontName=FONT_NAME,
            fontSize=17,
            leading=24,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),

        "option": ParagraphStyle(
            "option",
            fontName=FONT_NAME,
            fontSize=16,
            leading=22,
            leftIndent=7 * mm,
            spaceAfter=1.5 * mm,
        ),

        "answer": ParagraphStyle(
            "answer",
            fontName=FONT_NAME,
            fontSize=15,
            leading=21,
            leftIndent=5 * mm,
            spaceAfter=2 * mm,
        ),
    }


# ============================================================
# PDF HEADER / FOOTER
# ============================================================

def draw_page(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(
        colors.HexColor("#dddddd")
    )

    canvas.line(
        18 * mm,
        12 * mm,
        width - 18 * mm,
        12 * mm
    )

    canvas.setFont(
        FONT_NAME,
        10
    )

    canvas.setFillColor(
        colors.HexColor("#777777")
    )

    canvas.drawString(
        18 * mm,
        7 * mm,
        "AI Teacher V1.5"
    )

    canvas.drawRightString(
        width - 18 * mm,
        7 * mm,
        f"หน้า {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# BUILD PDF
# ============================================================

def build_pdf(data, output_file):

    styles = get_styles()

    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Teacher",
        author="AI Teacher",
    )

    story = []

    summary = data.get(
        "summary",
        {}
    )

    lesson_plan = data.get(
        "lesson_plan",
        {}
    )

    teaching = data.get(
        "teaching_content",
        {}
    )

    worksheet = data.get(
        "worksheet",
        []
    )

    quiz = data.get(
        "quiz",
        []
    )

    # ========================================================
    # COVER
    # ========================================================

    story.append(
        Paragraph(
            "ชุดการสอน",
            styles["title"]
        )
    )

    story.append(
        Paragraph(
            para_text(
                summary.get(
                    "topic",
                    "ชุดการเรียนรู้"
                )
            ),
            styles["subtitle"]
        )
    )

    meta = (
        f"วิชา {clean_text(summary.get('subject', ''))}"
        f" | ระดับชั้น {clean_text(summary.get('grade', ''))}"
        f" | เวลา {clean_text(summary.get('duration', ''))}"
    )

    if clean_text(summary.get("teacher_name")):
        meta += (
            f" | ครูผู้สอน "
            f"{clean_text(summary.get('teacher_name'))}"
        )

    story.append(
        Paragraph(
            para_text(meta),
            styles["small"]
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    # ========================================================
    # 1 LESSON PLAN
    # ========================================================

    story.append(
        Paragraph(
            "1. แผนการจัดการเรียนรู้",
            styles["heading"]
        )
    )

    story.append(
        Paragraph(
            "จุดประสงค์การเรียนรู้",
            styles["subheading"]
        )
    )

    for item in lesson_plan.get(
        "objective",
        []
    ):
        story.append(
            Paragraph(
                "• " + para_text(item),
                styles["body"]
            )
        )

    story.append(
        Paragraph(
            "ขั้นตอนการจัดการเรียนรู้",
            styles["subheading"]
        )
    )

    for step in lesson_plan.get(
        "steps",
        []
    ):

        time = clean_text(
            step.get("time", "")
        )

        title = clean_text(
            step.get("title", "")
        )

        detail = clean_text(
            step.get("detail", "")
        )

        block = []

        block.append(
            Paragraph(
                para_text(
                    f"{time} · {title}"
                ),
                styles["subheading"]
            )
        )

        block.append(
            Paragraph(
                para_text(detail),
                styles["body"]
            )
        )

        story.append(
            KeepTogether(block)
        )

    story.append(
        Paragraph(
            "การประเมินผล",
            styles["subheading"]
        )
    )

    story.append(
        Paragraph(
            para_text(
                lesson_plan.get(
                    "assessment",
                    ""
                )
            ),
            styles["body"]
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 2 TEACHING CONTENT
    # ========================================================

    story.append(
        Paragraph(
            "2. เนื้อหาที่ใช้สอน",
            styles["heading"]
        )
    )

    story.append(
        Paragraph(
            para_text(
                teaching.get(
                    "intro",
                    ""
                )
            ),
            styles["body"]
        )
    )

    story.append(
        Paragraph(
            "แนวคิดสำคัญ",
            styles["subheading"]
        )
    )

    for concept in teaching.get(
        "concepts",
        []
    ):

        story.append(
            Paragraph(
                "• " + para_text(concept),
                styles["body"]
            )
        )

    story.append(
        Paragraph(
            "ตัวอย่าง",
            styles["subheading"]
        )
    )

    for example in teaching.get(
        "examples",
        []
    ):

        story.append(
            Paragraph(
                para_text(
                    example.get(
                        "title",
                        ""
                    )
                ),
                styles["subheading"]
            )
        )

        story.append(
            Paragraph(
                para_text(
                    example.get(
                        "explanation",
                        ""
                    )
                ),
                styles["body"]
            )
        )

    story.append(
        Paragraph(
            "คำแนะนำสำหรับครู",
            styles["subheading"]
        )
    )

    for tip in teaching.get(
        "teacher_tips",
        []
    ):

        story.append(
            Paragraph(
                "• " + para_text(tip),
                styles["body"]
            )
        )

    story.append(
        Paragraph(
            "คำถามชวนคิด",
            styles["subheading"]
        )
    )

    for question in teaching.get(
        "thinking_questions",
        []
    ):

        story.append(
            Paragraph(
                "• " + para_text(question),
                styles["body"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 3 WORKSHEET
    # ========================================================

    story.append(
        Paragraph(
            "3. ใบงาน",
            styles["heading"]
        )
    )

    story.append(
        Paragraph(
            "ให้นักเรียนตอบคำถามต่อไปนี้",
            styles["body"]
        )
    )

    for index, item in enumerate(
        worksheet,
        start=1
    ):

        question = clean_text(
            item.get(
                "question",
                ""
            )
        )

        answer = clean_text(
            item.get(
                "answer",
                ""
            )
        )

        block = []

        block.append(
            Paragraph(
                para_text(
                    f"{index}. {question}"
                ),
                styles["question"]
            )
        )

        block.append(
            Spacer(
                1,
                5 * mm
            )
        )

        block.append(
            Paragraph(
                "คำตอบ: "
                + para_text(answer),
                styles["answer"]
            )
        )

        story.append(
            KeepTogether(block)
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 4 QUIZ
    # ========================================================

    story.append(
        Paragraph(
            "4. แบบทดสอบ",
            styles["heading"]
        )
    )

    story.append(
        Paragraph(
            "เลือกคำตอบที่ถูกต้องที่สุด",
            styles["body"]
        )
    )

    for index, item in enumerate(
        quiz,
        start=1
    ):

        question = clean_text(
            item.get(
                "question",
                ""
            )
        )

        options = item.get(
            "options",
            []
        )

        block = []

        block.append(
            Paragraph(
                para_text(
                    f"{index}. {question}"
                ),
                styles["question"]
            )
        )

        for option in options:

            block.append(
                Paragraph(
                    para_text(option),
                    styles["option"]
                )
            )

        story.append(
            KeepTogether(block)
        )

    # ========================================================
    # ANSWER KEY
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "5. เฉลย",
            styles["heading"]
        )
    )

    for index, item in enumerate(
        quiz,
        start=1
    ):

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
                para_text(
                    f"{index}. คำตอบ: {answer}"
                ),
                styles["question"]
            )
        )

        if explanation:

            story.append(
                Paragraph(
                    "อธิบาย: "
                    + para_text(
                        explanation
                    ),
                    styles["answer"]
                )
            )

    doc.build(
        story,
        onFirstPage=draw_page,
        onLaterPages=draw_page,
    )


# ============================================================
# OPENAI GENERATION
# ============================================================

def generate_lesson(req):

    if client is None:

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ OPENAI_API_KEY"
        )

    user_prompt = f"""
หัวข้อที่ต้องการ:
{req.prompt}

ชื่อครู:
{req.teacher_name or "ไม่ระบุ"}

จำนวนข้อแบบทดสอบ:
{req.question_count}

ระดับความยาก:
{req.difficulty}

สร้างชุดการสอนตาม SYSTEM PROMPT
และส่ง JSON ตาม Schema ที่กำหนดเท่านั้น
"""

    try:

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "teacher_pack",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"OpenAI error: {exc}"
        )

    try:

        result = json.loads(
            response.output_text
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="AI ส่งข้อมูล JSON ไม่ถูกต้อง"
        )

    return result


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ static/index.html"
        )

    return FileResponse(
        str(INDEX_FILE)
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "app": "AI Teacher",
        "version": "1.5.0",
        "model": MODEL,
        "font_regular": FONT_REGULAR.name,
        "font_regular_exists": FONT_REGULAR.exists(),
        "font_bolditalic": FONT_BOLDITALIC.name,
        "font_bolditalic_exists": FONT_BOLDITALIC.exists(),
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    data = generate_lesson(req)

    job_id = uuid.uuid4().hex

    job_dir = DATA_DIR / job_id

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_file = job_dir / "lesson.json"

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # สร้าง PDF รวมทั้งหมด
    # --------------------------------------------------------

    pdf_file = job_dir / "data.pdf"

    build_pdf(
        data,
        pdf_file
    )

    return {
        "ok": True,
        "job_id": job_id,
        "pdf_url": f"/api/pdf/{job_id}",
    }


# ============================================================
# PDF
# ============================================================

@app.get("/api/pdf/{job_id}")
def get_pdf(job_id: str):

    # ป้องกัน path traversal
    if not re.fullmatch(
        r"[a-f0-9]{32}",
        job_id
    ):

        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์ PDF"
        )

    pdf_file = (
        DATA_DIR
        / job_id
        / "data.pdf"
    )

    if not pdf_file.exists():

        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์ PDF"
        )

    # --------------------------------------------------------
    # สำคัญ:
    #
    # ไม่ใช้ Content-Disposition: attachment
    #
    # เพื่อให้ Safari เปิด PDF ให้ดูก่อน
    # แล้วผู้ใช้กดปุ่ม Download ของ Safari เอง
    # --------------------------------------------------------

    return FileResponse(
        path=str(pdf_file),
        media_type="application/pdf",
        filename="data.pdf",
        headers={
            "Content-Disposition":
                'inline; filename="data.pdf"'
        },
    )


# ============================================================
# CLEAN OLD DATA
# ============================================================

@app.post("/api/cleanup")
def cleanup():

    return {
        "ok": True
    }
