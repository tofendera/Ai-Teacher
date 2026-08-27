import os
import json
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# PATH
# =========================================================

APP_DIR = Path(__file__).resolve().parent

BASE_DIR = APP_DIR.parent

STATIC_DIR = BASE_DIR / "static"

INDEX_FILE = STATIC_DIR / "index.html"


# =========================================================
# FONT
# =========================================================
#
# โครงสร้าง:
#
# app/
# ├── main.py
# └── THSarabun.ttf
#

FONT_FILE = APP_DIR / "THSarabun.ttf"

FONT_NAME = "THSarabun"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.5.0"
)


# =========================================================
# MODEL
# =========================================================

MODEL = "gpt-5-mini"


# =========================================================
# QUESTION TYPE NAMES
# =========================================================

TYPE_NAMES = {

    "multiple_choice": "ปรนัย",

    "fill_blank": "เติมคำ",

    "calculation": "คำนวณ",

    "application": "ประยุกต์ใช้"
}


# =========================================================
# STATIC FILES
# =========================================================

if STATIC_DIR.exists():

    app.mount(

        "/static",

        StaticFiles(
            directory=str(STATIC_DIR)
        ),

        name="static"
    )


# =========================================================
# REQUEST MODEL
# =========================================================

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

    difficulty: str = Field(
        default="mixed",
        max_length=30
    )


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """

คุณคือผู้ช่วยจัดทำเอกสารการเรียนการสอนสำหรับครูไทย

หน้าที่คือเปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็นชุดเอกสารการเรียนการสอนพร้อมใช้งาน

ตัวอย่าง:

"เศษส่วน ป.4 1 ชั่วโมง"

ให้สร้าง:

1. ข้อมูลบทเรียน
2. จุดประสงค์การเรียนรู้
3. เนื้อหาที่ใช้สอน
4. ตัวอย่างสำหรับใช้สอน
5. คำถามชวนคิด
6. ขั้นตอนการจัดการเรียนรู้
7. การประเมินผล
8. ใบงาน
9. เฉลยใบงาน
10. แบบทดสอบ
11. เฉลยแบบทดสอบ


หลักการ:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้น
- วิเคราะห์วิชา
- วิเคราะห์หัวข้อ
- วิเคราะห์เวลาเรียน
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น
- หากข้อมูลบางส่วนไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะถ้าไม่แน่ใจ
- ห้ามสร้างข้อมูลมั่ว
- ตรวจสอบคำตอบก่อนส่ง


เนื้อหาที่ใช้สอน:

ต้องเขียนให้ครูสามารถนำไปอธิบายกับนักเรียนได้จริง

ควรมี:

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบาย
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ใบงาน:

- เป็นแบบฝึกหัดให้นักเรียน
- เหมาะกับระดับชั้น
- มีคำถามชัดเจน
- สอดคล้องกับเนื้อหา
- มีคำตอบสำหรับใช้เป็นเฉลย


แบบทดสอบ:

ต้องสร้างเฉพาะประเภทที่ผู้ใช้เลือก

ประเภทที่รองรับ:

multiple_choice
fill_blank
calculation
application


ปรนัย:

- 4 ตัวเลือก
- มีคำตอบถูกเพียงหนึ่งข้อ


เติมคำ:

- ต้องมีคำตอบที่ชัดเจน


คำนวณ:

- ตรวจสอบตัวเลข
- ตรวจสอบหน่วย
- ตรวจสอบคำตอบ


ประยุกต์ใช้:

- เป็นสถานการณ์ที่เหมาะกับวัย
- ต้องสัมพันธ์กับเนื้อหา


ห้ามสร้างประเภทข้อสอบที่ผู้ใช้ไม่ได้เลือก


ตอบเป็น JSON ตาม Schema เท่านั้น

"""


# =========================================================
# JSON SCHEMA
# =========================================================

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
                }

            },

            "required": [
                "grade",
                "subject",
                "topic",
                "duration"
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


# =========================================================
# FONT REGISTRATION
# =========================================================

def register_thai_font():

    if not FONT_FILE.exists():

        raise FileNotFoundError(

            "ไม่พบไฟล์ Font: "
            f"{FONT_FILE}"

        )


    registered_fonts = (
        pdfmetrics
        .getRegisteredFontNames()
    )


    if FONT_NAME not in registered_fonts:

        pdfmetrics.registerFont(

            TTFont(

                FONT_NAME,

                str(FONT_FILE)

            )

        )


    return FONT_NAME


# =========================================================
# ESCAPE
# =========================================================

def esc(value):

    return (

        str(value or "")

        .replace("&", "&amp;")

        .replace("<", "&lt;")

        .replace(">", "&gt;")

    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        raise HTTPException(

            status_code=500,

            detail=
            "ไม่พบไฟล์ static/index.html"

        )


    return FileResponse(

        str(INDEX_FILE),

        media_type="text/html"

    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "app": "Teacher Pack",

        "version": "1.5.0",

        "model": MODEL,

        "font": FONT_FILE.name,

        "font_path": str(FONT_FILE),

        "font_exists": FONT_FILE.exists()

    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/api/generate")
def generate(
    req: GenerateRequest
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )


    if not api_key:

        raise HTTPException(

            status_code=500,

            detail=
            "ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render"

        )


    allowed_types = {

        "multiple_choice",

        "fill_blank",

        "calculation",

        "application"

    }


    if not req.question_types:

        raise HTTPException(

            status_code=400,

            detail=
            "กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ"

        )


    invalid_types = [

        item

        for item in req.question_types

        if item not in allowed_types

    ]


    if invalid_types:

        raise HTTPException(

            status_code=400,

            detail=
            "รูปแบบข้อสอบไม่ถูกต้อง: "
            + ", ".join(invalid_types)

        )


    selected_types = ", ".join(

        TYPE_NAMES[item]

        for item in req.question_types

    )


    teacher_name = (
        req.teacher_name.strip()
        or "ไม่ได้ระบุ"
    )


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


ชื่อครูผู้สอน:

{teacher_name}


จำนวนข้อสอบ:

{req.question_count}


รูปแบบข้อสอบที่เลือก:

{selected_types}


ระดับความยาก:

{req.difficulty}


สร้าง Teacher Pack ครบชุด

สร้างเนื้อหาที่ครูสามารถนำไปสอนได้จริง

สร้างใบงานประมาณ 10-20 ข้อ

สร้างแบบทดสอบจำนวน
{req.question_count} ข้อ

แบบทดสอบต้องใช้เฉพาะประเภทที่เลือก

ห้ามสร้างประเภทที่ไม่ได้เลือก

ตรวจสอบคำตอบทุกข้อ

ตรวจสอบตัวเลขและหน่วย

ตรวจสอบความสอดคล้องระหว่างคำถามกับเฉลย

"""


    try:

        client = OpenAI(
            api_key=api_key
        )


        response = client.responses.create(

            model=MODEL,

            instructions=SYSTEM_PROMPT,

            input=user_prompt,

            text={

                "format": {

                    "type": "json_schema",

                    "name": "teacher_pack",

                    "strict": True,

                    "schema": SCHEMA

                }

            }

        )


        output_text = (
            response.output_text
        )


        if not output_text:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        data = json.loads(
            output_text
        )


        data["_teacher_name"] = (
            teacher_name
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail=
            "AI ส่งข้อมูลกลับมาไม่ใช่ JSON ที่ถูกต้อง"

        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
            "สร้างชุดการสอนไม่สำเร็จ: "
            + str(e)

        )


# =========================================================
# PDF STYLES
# =========================================================

def create_pdf_styles(font_name):

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "TitleTH",

        parent=styles["Title"],

        fontName=font_name,

        fontSize=20,

        leading=25,

        alignment=TA_CENTER,

        spaceAfter=8

    )


    subtitle = ParagraphStyle(

        "SubtitleTH",

        parent=styles["BodyText"],

        fontName=font_name,

        fontSize=13,

        leading=18,

        alignment=TA_CENTER,

        spaceAfter=10

    )


    h1 = ParagraphStyle(

        "H1TH",

        parent=styles["Heading1"],

        fontName=font_name,

        fontSize=16,

        leading=21,

        spaceBefore=10,

        spaceAfter=7

    )


    h2 = ParagraphStyle(

        "H2TH",

        parent=styles["Heading2"],

        fontName=font_name,

        fontSize=13,

        leading=18,

        spaceBefore=8,

        spaceAfter=5

    )


    body = ParagraphStyle(

        "BodyTH",

        parent=styles["BodyText"],

        fontName=font_name,

        fontSize=14,

        leading=21,

        alignment=TA_LEFT,

        spaceAfter=6

    )


    table = ParagraphStyle(

        "TableTH",

        parent=body,

        fontSize=11,

        leading=16

    )


    question = ParagraphStyle(

        "QuestionTH",

        parent=body,

        fontSize=14,

        leading=21,

        spaceAfter=5

    )


    return {

        "title": title,

        "subtitle": subtitle,

        "h1": h1,

        "h2": h2,

        "body": body,

        "table": table,

        "question": question

    }


# =========================================================
# INFO TABLE
# =========================================================

def create_info_table(
    summary,
    teacher_name,
    styles
):

    teacher_name = (
        teacher_name.strip()
        or "ไม่ได้ระบุ"
    )


    rows = [

        [

            Paragraph(
                "<b>วิชา</b><br/>"
                + esc(
                    summary.get(
                        "subject",
                        ""
                    )
                ),
                styles["table"]
            ),

            Paragraph(
                "<b>ระดับชั้น</b><br/>"
                + esc(
                    summary.get(
                        "grade",
                        ""
                    )
                ),
                styles["table"]
            )

        ],

        [

            Paragraph(
                "<b>เรื่อง</b><br/>"
                + esc(
                    summary.get(
                        "topic",
                        ""
                    )
                ),
                styles["table"]
            ),

            Paragraph(
                "<b>เวลา</b><br/>"
                + esc(
                    summary.get(
                        "duration",
                        ""
                    )
                ),
                styles["table"]
            )

        ],

        [

            Paragraph(
                "<b>ครูผู้สอน</b><br/>"
                + esc(
                    teacher_name
                ),
                styles["table"]
            ),

            Paragraph(
                "<b>วันที่</b><br/>"
                "________________________",
                styles["table"]
            )

        ]

    ]


    table = Table(

        rows,

        colWidths=[
            82 * mm,
            82 * mm
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#bdbdbd")
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f7f5ff")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    return table


# =========================================================
# PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    font_name = register_thai_font()

    styles = create_pdf_styles(
        font_name
    )


    buffer = BytesIO()


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        leftMargin=18 * mm,

        rightMargin=18 * mm,

        topMargin=17 * mm,

        bottomMargin=18 * mm,

        title="เอกสารการเรียนการสอน"

    )


    story = []


    summary = data.get(
        "summary",
        {}
    )


    teacher_name = data.get(
        "_teacher_name",
        ""
    )


    lesson = data.get(
        "lesson_plan",
        {}
    )


    teaching = data.get(
        "teaching_content",
        {}
    )


    # =====================================================
    # COMMON HEADER
    # =====================================================

    def add_header(title):

        story.append(

            Paragraph(
                title,
                styles["title"]
            )

        )


        story.append(

            Paragraph(

                "เรื่อง "
                + esc(
                    summary.get(
                        "topic",
                        ""
                    )
                ),

                styles["subtitle"]

            )

        )


        story.append(

            create_info_table(

                summary,

                teacher_name,

                styles

            )

        )


        story.append(
            Spacer(
                1,
                7 * mm
            )
        )


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        add_header(
            "แผนการจัดการเรียนรู้"
        )


        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                styles["h1"]
            )

        )


        for i, item in enumerate(

            lesson.get(
                "objective",
                []
            ),

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body"]

                )

            )


        story.append(

            Paragraph(
                "2. เนื้อหาที่ใช้สอน",
                styles["h1"]
            )

        )


        story.append(

            Paragraph(

                esc(
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
                "2.1 สาระสำคัญ",
                styles["h2"]
            )

        )


        for i, item in enumerate(

            teaching.get(
                "concepts",
                []
            ),

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body"]

                )

            )


        story.append(

            Paragraph(
                "2.2 ตัวอย่างสำหรับใช้สอน",
                styles["h2"]
            )

        )


        for i, item in enumerate(

            teaching.get(
                "examples",
                []
            ),

            start=1

        ):

            story.append(

                KeepTogether([

                    Paragraph(

                        f"ตัวอย่างที่ {i} "
                        f"{esc(item.get('title',''))}",

                        styles["question"]

                    ),

                    Paragraph(

                        esc(
                            item.get(
                                "explanation",
                                ""
                            )
                        ),

                        styles["body"]

                    )

                ])

            )


        story.append(

            Paragraph(
                "2.3 คำถามชวนคิด",
                styles["h2"]
            )

        )


        for i, item in enumerate(

            teaching.get(
                "thinking_questions",
                []
            ),

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body"]

                )

            )


        story.append(

            Paragraph(
                "3. ขั้นตอนการจัดการเรียนรู้",
                styles["h1"]
            )

        )


        rows = [

            [

                Paragraph(
                    "<b>เวลา</b>",
                    styles["table"]
                ),

                Paragraph(
                    "<b>กิจกรรม</b>",
                    styles["table"]
                ),

                Paragraph(
                    "<b>รายละเอียด</b>",
                    styles["table"]
                )

            ]

        ]


        for item in lesson.get(
            "steps",
            []
        ):

            rows.append(

                [

                    Paragraph(
                        esc(
                            item.get(
                                "time",
                                ""
                            )
                        ),
                        styles["table"]
                    ),

                    Paragraph(
                        esc(
                            item.get(
                                "title",
                                ""
                            )
                        ),
                        styles["table"]
                    ),

                    Paragraph(
                        esc(
                            item.get(
                                "detail",
                                ""
                            )
                        ),
                        styles["table"]
                    )

                ]

            )


        if len(rows) > 1:

            table = Table(

                rows,

                colWidths=[
                    25 * mm,
                    42 * mm,
                    99 * mm
                ],

                repeatRows=1

            )


            table.setStyle(

                TableStyle([

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#cfcfcf")
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#eeeaff")
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )

                ])

            )


            story.append(
                table
            )


        story.append(

            Paragraph(
                "4. เคล็ดลับสำหรับครู",
                styles["h1"]
            )

        )


        for i, item in enumerate(

            teaching.get(
                "teacher_tips",
                []
            ),

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body"]

                )

            )


        story.append(

            Paragraph(
                "5. การประเมินผล",
                styles["h1"]
            )

        )


        story.append(

            Paragraph(

                esc(
                    lesson.get(
                        "assessment",
                        ""
                    )
                ),

                styles["body"]

            )

        )


    # =====================================================
    # WORKSHEET
    # =====================================================

    def add_worksheet():

        story.append(
            PageBreak()
        )


        add_header(
            "ใบงาน"
        )


        story.append(

            Paragraph(

                "ชื่อ-สกุล "
                ".................................................................",

                styles["body"]

            )

        )


        story.append(

            Paragraph(

                "ชั้น ................ "
                "เลขที่ ................ "
                "วันที่ ................",

                styles["body"]

            )

        )


        story.append(

            Paragraph(
                "คำชี้แจง",
                styles["h2"]
            )

        )


        story.append(

            Paragraph(

                "ให้นักเรียนอ่านคำถาม "
                "และเขียนคำตอบลงในพื้นที่ที่กำหนด",

                styles["body"]

            )

        )


        for item in data.get(
            "worksheet",
            []
        ):

            block = [

                Paragraph(

                    f"<b>ข้อ {item.get('no','')}.</b> "
                    f"{esc(item.get('question',''))}",

                    styles["question"]

                ),

                Paragraph(

                    "คำตอบ "
                    "................................................................................",

                    styles["body"]

                ),

                Paragraph(

                    "................................................................................",

                    styles["body"]

                )

            ]


            story.append(

                KeepTogether(
                    block
                )

            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(
            PageBreak()
        )


        add_header(
            "แบบทดสอบ"
        )


        story.append(

            Paragraph(

                "ชื่อ-สกุล "
                ".................................................................",

                styles["body"]

            )

        )


        story.append(

            Paragraph(

                "ชั้น ................ "
                "เลขที่ ................",

                styles["body"]

            )

        )


        story.append(

            Paragraph(

                "คำชี้แจง: "
                "ทำแบบทดสอบทุกข้อ",

                styles["body"]

            )

        )


        for item in data.get(
            "quiz",
            []
        ):

            block = []


            type_name = TYPE_NAMES.get(

                item.get(
                    "type",
                    ""
                ),

                item.get(
                    "type",
                    ""
                )

            )


            block.append(

                Paragraph(

                    f"<b>ข้อ {item.get('no','')} "
                    f"({esc(type_name)})</b>",

                    styles["question"]

                )

            )


            block.append(

                Paragraph(

                    esc(
                        item.get(
                            "question",
                            ""
                        )
                    ),

                    styles["body"]

                )

            )


            options = item.get(
                "options",
                []
            )


            if options:

                for i, option in enumerate(
                    options
                ):

                    letter = chr(
                        65 + i
                    )


                    block.append(

                        Paragraph(

                            f"{letter}. "
                            f"{esc(option)}",

                            styles["body"]

                        )

                    )

            else:

                block.append(

                    Paragraph(

                        "คำตอบ "
                        "................................................",

                        styles["body"]

                    )

                )


            story.append(

                KeepTogether(
                    block
                )

            )


    # =====================================================
    # ANSWERS
    # =====================================================

    def add_answers():

        story.append(
            PageBreak()
        )


        add_header(
            "เฉลย"
        )


        story.append(

            Paragraph(
                "เฉลยใบงาน",
                styles["h1"]
            )

        )


        for item in data.get(
            "worksheet",
            []
        ):

            story.append(

                Paragraph(

                    f"ข้อ {item.get('no','')}. "
                    f"{esc(item.get('answer',''))}",

                    styles["body"]

                )

            )


        story.append(

            Paragraph(
                "เฉลยแบบทดสอบ",
                styles["h1"]
            )

        )


        for item in data.get(
            "quiz",
            []
        ):

            story.append(

                KeepTogether([

                    Paragraph(

                        f"ข้อ {item.get('no','')}. "
                        f"{esc(item.get('answer',''))}",

                        styles["question"]

                    ),

                    Paragraph(

                        esc(
                            item.get(
                                "explanation",
                                ""
                            )
                        ),

                        styles["body"]

                    )

                ])

            )


    # =====================================================
    # SECTION
    # =====================================================

    if section in (
        "all",
        "lesson"
    ):

        add_lesson()


    if section in (
        "all",
        "worksheet"
    ):

        add_worksheet()


    if section in (
        "all",
        "quiz"
    ):

        add_quiz()


    if section in (
        "all",
        "answers"
    ):

        add_answers()


    # =====================================================
    # BUILD
    # =====================================================

    doc.build(
        story
    )


    buffer.seek(0)

    return buffer


# =========================================================
# PDF ENDPOINT
# =========================================================

@app.post("/api/pdf")
def create_pdf(
    data: dict,
    section: str = "all"
):

    allowed_sections = {

        "all",
        "lesson",
        "worksheet",
        "quiz",
        "answers"

    }


    if section not in allowed_sections:

        raise HTTPException(

            status_code=400,

            detail="section ไม่ถูกต้อง"

        )


    try:

        pdf_file = build_pdf(

            data,

            section

        )


        filename = (
            f"teacher-pack-{section}.pdf"
        )


        return StreamingResponse(

            pdf_file,

            media_type="application/pdf",

            headers={

                "Content-Disposition":
                f'attachment; filename="{filename}"'

            }

        )


    except FileNotFoundError as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "สร้าง PDF ไม่สำเร็จ: "
                + str(e)
            )

        )
