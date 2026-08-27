import os
import json
import uuid
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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
FONT_BOLDITALIC = APP_DIR / "THSarabun BoldItalic.ttf"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Teacher Pack V1.0",
    version="1.0.0",
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

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini",
)

client = (
    OpenAI()
    if os.getenv("OPENAI_API_KEY")
    else None
)


# ============================================================
# TEMPORARY MEMORY
# ============================================================

JOBS: dict[str, dict[str, Any]] = {}

JOB_TTL = 60 * 60


# ============================================================
# REQUEST
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        min_length=2,
        max_length=500,
    )

    teacher_name: str = Field(
        default="",
        max_length=100,
    )

    question_count: int = Field(
        default=10,
        ge=5,
        le=30,
    )

    difficulty: str = Field(
        default="mixed",
        max_length=30,
    )


# ============================================================
# JSON SCHEMA
# ============================================================

SCHEMA = {

    "type": "object",

    "additionalProperties": False,

    "properties": {

        "title": {
            "type": "string"
        },

        "grade": {
            "type": "string"
        },

        "duration": {
            "type": "string"
        },

        "objectives": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "lesson_plan": {

            "type": "array",

            "items": {

                "type": "object",

                "additionalProperties": False,

                "properties": {

                    "stage": {
                        "type": "string"
                    },

                    "minutes": {
                        "type": "string"
                    },

                    "activity": {
                        "type": "string"
                    },
                },

                "required": [
                    "stage",
                    "minutes",
                    "activity",
                ],
            },
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
                            },
                        },

                        "required": [
                            "title",
                            "explanation",
                        ],
                    },
                },

                "thinking_questions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
            },

            "required": [
                "intro",
                "concepts",
                "examples",
                "thinking_questions",
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
                },

                "required": [
                    "no",
                    "question",
                    "answer",
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

                    "options": {

                        "type": "array",

                        "items": {
                            "type": "string"
                        },
                    },

                    "answer": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    },
                },

                "required": [
                    "no",
                    "question",
                    "options",
                    "answer",
                    "explanation",
                ],
            },
        },

        "answer_key": {

            "type": "array",

            "items": {

                "type": "object",

                "additionalProperties": False,

                "properties": {

                    "no": {
                        "type": "integer"
                    },

                    "answer": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    },
                },

                "required": [
                    "no",
                    "answer",
                    "explanation",
                ],
            },
        },
    },

    "required": [
        "title",
        "grade",
        "duration",
        "objectives",
        "lesson_plan",
        "teaching_content",
        "worksheet",
        "quiz",
        "answer_key",
    ],
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
คุณคือผู้ช่วยจัดทำชุดการสอนสำหรับครูไทย

สร้างชุดการสอนภาษาไทยที่พร้อมนำไปใช้จริง
และต้องส่งข้อมูลตาม JSON Schema เท่านั้น

ข้อกำหนด:

- ใช้ภาษาไทยเป็นหลัก
- ห้ามใส่ emoji ในเนื้อหาที่สร้าง
- ห้ามใช้คำว่า "ครูผู้ช่วย" ทุกกรณี
- ห้ามใส่คำว่า "(ปรนัย)" ต่อท้ายคำถาม
- แบบทดสอบเป็นคำถาม 4 ตัวเลือก
- ตัวเลือกใช้ ก. ข. ค. ง.
- มีคำตอบถูกเพียง 1 ตัวเลือก
- ตรวจสอบตัวเลขและการคำนวณให้ถูกต้อง
- ตรวจสอบคำตอบทุกข้อ
- เลขข้อเริ่มจาก 1 และเรียงต่อเนื่อง
- ใน quiz ช่อง question ต้องเป็นคำถามล้วน
- ห้ามใส่เลขข้อซ้ำใน question
- answer ของ quiz ต้องเป็นตัวเลือกที่ถูก เช่น "ก."
- answer_key ต้องตรงกับ quiz ทุกข้อ
- เนื้อหาต้องเหมาะกับระดับชั้น
- เนื้อหาต้องตรงกับหัวข้อที่ผู้ใช้ระบุ
"""


# ============================================================
# CLEANUP
# ============================================================

def cleanup_jobs():

    now = time.time()

    for job_id in list(JOBS):

        if (
            now - JOBS[job_id]["created_at"]
            > JOB_TTL
        ):

            del JOBS[job_id]


# ============================================================
# REMOVE UNWANTED TEXT
# ============================================================

def clean_text(value: Any) -> Any:

    if isinstance(value, str):

        return (
            value
            .replace("ครูผู้ช่วย", "")
            .replace("(ปรนัย)", "")
            .strip()
        )

    if isinstance(value, list):

        return [
            clean_text(item)
            for item in value
        ]

    if isinstance(value, dict):

        return {
            key: clean_text(val)
            for key, val in value.items()
        }

    return value


# ============================================================
# PDF FONT
# ============================================================

def register_fonts():

    regular = "Helvetica"
    bold = "Helvetica-Bold"

    if FONT_REGULAR.exists():

        try:

            pdfmetrics.registerFont(
                TTFont(
                    "THSarabun",
                    str(FONT_REGULAR),
                )
            )

            regular = "THSarabun"

        except Exception:
            pass

    if FONT_BOLDITALIC.exists():

        try:

            pdfmetrics.registerFont(
                TTFont(
                    "THSarabunBoldItalic",
                    str(FONT_BOLDITALIC),
                )
            )

            bold = "THSarabunBoldItalic"

        except Exception:
            pass

    return regular, bold


# ============================================================
# PDF TEXT
# ============================================================

def safe_pdf_text(value):

    import html

    return html.escape(
        str(value or "")
    )


# ============================================================
# BUILD ONE PDF
# ============================================================

def build_pdf(
    data: dict[str, Any],
    teacher_name: str,
) -> bytes:

    regular, bold = register_fonts()

    buffer = BytesIO()

    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,
        leftMargin=18 * mm,

        topMargin=18 * mm,
        bottomMargin=18 * mm,

        title="Teacher Pack",

        author=(
            teacher_name
            or "Teacher Pack"
        ),
    )

    title_style = ParagraphStyle(

        "Title",

        fontName=bold,

        fontSize=24,

        leading=30,

        alignment=TA_CENTER,

        spaceAfter=10,
    )

    heading_style = ParagraphStyle(

        "Heading",

        fontName=bold,

        fontSize=19,

        leading=24,

        spaceBefore=10,

        spaceAfter=8,
    )

    subheading_style = ParagraphStyle(

        "SubHeading",

        fontName=bold,

        fontSize=16,

        leading=21,

        spaceBefore=8,

        spaceAfter=5,
    )

    body_style = ParagraphStyle(

        "Body",

        fontName=regular,

        fontSize=14.5,

        leading=20,

        spaceAfter=5,
    )

    small_style = ParagraphStyle(

        "Small",

        fontName=regular,

        fontSize=12.5,

        leading=17,

        textColor=colors.HexColor(
            "#555555"
        ),
    )

    story = []


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(

        Paragraph(

            safe_pdf_text(
                data.get(
                    "title",
                    "ชุดการสอน"
                )
            ),

            title_style,
        )
    )


    # --------------------------------------------------------
    # META
    # --------------------------------------------------------

    meta = []

    if teacher_name.strip():

        meta.append(
            f"ชื่อครู: {teacher_name.strip()}"
        )

    if data.get("grade"):

        meta.append(
            f"ระดับชั้น: {data['grade']}"
        )

    if data.get("duration"):

        meta.append(
            f"เวลา: {data['duration']}"
        )

    if meta:

        story.append(

            Paragraph(

                safe_pdf_text(
                    " | ".join(meta)
                ),

                small_style,
            )
        )

    story.append(
        Spacer(1, 8)
    )


    # --------------------------------------------------------
    # 1 OBJECTIVES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. จุดประสงค์การเรียนรู้",
            heading_style,
        )
    )

    for index, item in enumerate(
        data.get("objectives", []),
        1,
    ):

        story.append(

            Paragraph(

                safe_pdf_text(
                    f"{index}. {item}"
                ),

                body_style,
            )
        )


    # --------------------------------------------------------
    # 2 LESSON PLAN
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. แผนการจัดการเรียนรู้",
            heading_style,
        )
    )

    for item in data.get(
        "lesson_plan",
        [],
    ):

        story.append(

            KeepTogether(

                [

                    Paragraph(

                        safe_pdf_text(

                            f"{item.get('stage', '')}"
                            f" — "
                            f"{item.get('minutes', '')}"

                        ),

                        subheading_style,
                    ),

                    Paragraph(

                        safe_pdf_text(
                            item.get(
                                "activity",
                                "",
                            )
                        ),

                        body_style,
                    ),
                ]
            )
        )


    # --------------------------------------------------------
    # 3 TEACHING CONTENT
    # --------------------------------------------------------

    content = data.get(
        "teaching_content",
        {},
    )

    story.append(
        Paragraph(
            "3. เนื้อหาที่ใช้สอน",
            heading_style,
        )
    )

    story.append(

        Paragraph(

            safe_pdf_text(
                content.get(
                    "intro",
                    "",
                )
            ),

            body_style,
        )
    )

    for concept in content.get(
        "concepts",
        [],
    ):

        story.append(

            Paragraph(

                safe_pdf_text(
                    f"• {concept}"
                ),

                body_style,
            )
        )

    for example in content.get(
        "examples",
        [],
    ):

        story.append(

            Paragraph(

                safe_pdf_text(
                    example.get(
                        "title",
                        "ตัวอย่าง",
                    )
                ),

                subheading_style,
            )
        )

        story.append(

            Paragraph(

                safe_pdf_text(
                    example.get(
                        "explanation",
                        "",
                    )
                ),

                body_style,
            )
        )


    # --------------------------------------------------------
    # THINKING QUESTIONS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "คำถามชวนคิด",
            subheading_style,
        )
    )

    for index, question in enumerate(
        content.get(
            "thinking_questions",
            [],
        ),
        1,
    ):

        story.append(

            Paragraph(

                safe_pdf_text(
                    f"{index}. {question}"
                ),

                body_style,
            )
        )


    # --------------------------------------------------------
    # WORKSHEET
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. ใบงาน",
            heading_style,
        )
    )

    for item in data.get(
        "worksheet",
        [],
    ):

        story.append(

            Paragraph(

                safe_pdf_text(

                    f"{item.get('no', '')}. "
                    f"{item.get('question', '')}"

                ),

                body_style,
            )
        )

        story.append(

            Paragraph(

                "คำตอบ: "
                "______________________________________________",

                body_style,
            )
        )


    # --------------------------------------------------------
    # NEW PAGE
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )


    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. แบบทดสอบ",
            heading_style,
        )
    )

    for item in data.get(
        "quiz",
        [],
    ):

        block = [

            Paragraph(

                safe_pdf_text(

                    f"{item.get('no', '')}. "
                    f"{item.get('question', '')}"

                ),

                subheading_style,
            )
        ]

        for option in item.get(
            "options",
            [],
        ):

            block.append(

                Paragraph(

                    safe_pdf_text(
                        option
                    ),

                    body_style,
                )
            )

        story.append(
            KeepTogether(block)
        )


    # --------------------------------------------------------
    # ANSWER KEY PAGE
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "6. เฉลย",
            heading_style,
        )
    )

    for item in data.get(
        "answer_key",
        [],
    ):

        story.append(

            Paragraph(

                safe_pdf_text(

                    f"{item.get('no', '')}. "
                    f"{item.get('answer', '')}"

                ),

                subheading_style,
            )
        )

        story.append(

            Paragraph(

                safe_pdf_text(
                    item.get(
                        "explanation",
                        "",
                    )
                ),

                body_style,
            )
        )


    doc.build(story)

    return buffer.getvalue()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        raise HTTPException(

            status_code=500,

            detail="ไม่พบ static/index.html",
        )

    return FileResponse(

        str(INDEX_FILE),

        headers={
            "Cache-Control":
            "no-store"
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "app": "Teacher Pack",

        "version": "1.0.0",

        "openai_configured":
            client is not None,

        "font_regular_exists":
            FONT_REGULAR.exists(),

        "font_bolditalic_exists":
            FONT_BOLDITALIC.exists(),
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/api/generate")
def generate(
    req: GenerateRequest,
):

    cleanup_jobs()


    if client is None:

        raise HTTPException(

            status_code=500,

            detail=
            "ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render",
        )


    user_prompt = f"""

หัวข้อที่ต้องการสอน:
{req.prompt}

ชื่อครู:
{req.teacher_name or "ไม่ระบุ"}

จำนวนข้อสอบ:
{req.question_count}

ระดับความยาก:
{req.difficulty}

สร้างชุดการสอนให้ครบทุกส่วนตาม schema

แบบทดสอบต้องมีจำนวน
{req.question_count}
ข้อพอดี

"""


    try:

        response = client.responses.create(

            model=MODEL,

            input=[

                {
                    "role": "system",

                    "content": [
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT,
                        }
                    ],
                },

                {
                    "role": "user",

                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },

            ],

            text={

                "format": {

                    "type":
                        "json_schema",

                    "name":
                        "teacher_pack",

                    "strict":
                        True,

                    "schema":
                        SCHEMA,
                }
            },
        )


        data = json.loads(
            response.output_text
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=
            f"สร้างชุดการสอนไม่สำเร็จ: {exc}",
        )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    data = clean_text(data)


    # --------------------------------------------------------
    # CHECK QUIZ
    # --------------------------------------------------------

    quiz = data.get(
        "quiz",
        [],
    )

    if len(quiz) != req.question_count:

        raise HTTPException(

            status_code=500,

            detail=
            "AI สร้างจำนวนข้อสอบไม่ครบ กรุณากดสร้างใหม่",
        )


    # --------------------------------------------------------
    # FORCE NUMBERING
    # --------------------------------------------------------

    for index, item in enumerate(
        quiz,
        1,
    ):

        item["no"] = index


    for index, item in enumerate(
        data.get(
            "answer_key",
            [],
        ),
        1,
    ):

        item["no"] = index


    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex


    # --------------------------------------------------------
    # CREATE ONE PDF
    # --------------------------------------------------------

    pdf_bytes = build_pdf(
        data,
        req.teacher_name,
    )


    # --------------------------------------------------------
    # STORE TEMPORARILY
    # --------------------------------------------------------

    JOBS[job_id] = {

        "created_at":
            time.time(),

        "data":
            data,

        "pdf":
            pdf_bytes,
    }


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "job_id":
            job_id,

        "data":
            data,

        "pdf_url":
            f"/api/pdf/{job_id}/data.pdf",
    }


# ============================================================
# PDF
# ============================================================

@app.get(
    "/api/pdf/{job_id}/data.pdf"
)
def get_pdf(
    job_id: str,
):

    cleanup_jobs()


    job = JOBS.get(
        job_id
    )


    if not job:

        raise HTTPException(

            status_code=404,

            detail=
            "ไม่พบไฟล์ PDF ของชุดการสอนนี้",
        )


    return Response(

        content=job["pdf"],

        media_type=
            "application/pdf",

        headers={

            "Content-Disposition":
                'inline; filename="data.pdf"',

            "Cache-Control":
                "no-store, max-age=0",
        },
    )
