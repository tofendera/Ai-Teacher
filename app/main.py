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
# สำคัญ:
# ฟอนต์อยู่ใน app/
#
# app/
# ├── main.py
# └── ANGSA.ttf
#

FONT_FILE = APP_DIR / "ANGSA.ttf"

FONT_NAME = "AngsanaNew"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.5.0"
)


# =========================================================
# TYPE NAMES
# =========================================================

TYPE_NAMES = {
    "multiple_choice": "ปรนัย",
    "fill_blank": "เติมคำ",
    "calculation": "คำนวณ",
    "application": "ประยุกต์ใช้"
}


# =========================================================
# STATIC
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
# OPENAI
# =========================================================

MODEL = "gpt-5-mini"


# =========================================================
# REQUEST MODEL
# =========================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        min_length=2,
        max_length=500
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

    teacher_name: str = Field(
        default="",
        max_length=100
    )


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """

คุณคือผู้ช่วยจัดทำเอกสารการสอนสำหรับครูไทย

หน้าที่คือเปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็นชุดการสอนพร้อมใช้งาน

ชุดการสอนประกอบด้วย:

1. ข้อมูลสรุปบทเรียน
2. จุดประสงค์การเรียนรู้
3. เนื้อหาที่ใช้สอน
4. ตัวอย่างสำหรับอธิบาย
5. คำถามชวนคิด
6. ขั้นตอนการสอน
7. ใบงาน
8. เฉลยใบงาน
9. แบบทดสอบ
10. เฉลยแบบทดสอบ

กติกา:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้น
- วิเคราะห์วิชา
- วิเคราะห์หัวข้อ
- วิเคราะห์เวลา
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น
- หากข้อมูลไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะถ้าไม่แน่ใจ
- ห้ามสร้างข้อมูลมั่ว
- ตรวจสอบความถูกต้องก่อนส่ง

เนื้อหาการสอนต้องสามารถนำไปใช้จริงได้

ใบงานต้องเป็นแบบฝึกหัดสำหรับนักเรียน

แบบทดสอบต้องสร้างเฉพาะประเภทที่ผู้ใช้เลือก

ประเภทที่รองรับ:

multiple_choice
fill_blank
calculation
application

ปรนัย:

- 4 ตัวเลือก
- มีคำตอบถูกเพียงหนึ่งข้อ

เติมคำ:

- มีคำตอบชัดเจน

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
# REGISTER FONT
# =========================================================

def register_font():

    if not FONT_FILE.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_FILE}"
        )

    try:

        pdfmetrics.registerFont(
            TTFont(
                FONT_NAME,
                str(FONT_FILE)
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"โหลด Font ไม่สำเร็จ: {e}"
        )

    return FONT_NAME


# =========================================================
# ESCAPE
# =========================================================

def esc(text):

    return (
        str(text or "")
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
            detail="ไม่พบไฟล์ static/index.html"
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

        "font_exists": FONT_FILE.exists()
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render"
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
            detail="กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ"
        )


    invalid_types = [

        t

        for t in req.question_types

        if t not in allowed_types
    ]


    if invalid_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "รูปแบบข้อสอบไม่ถูกต้อง: "
                + ", ".join(invalid_types)
            )
        )


    selected_types = ", ".join(

        TYPE_NAMES[t]

        for t in req.question_types
    )


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


จำนวนข้อสอบ:

{req.question_count} ข้อ


รูปแบบข้อสอบที่เลือก:

{selected_types}


ระดับความยาก:

{req.difficulty}


ชื่อครู:

{req.teacher_name or "ไม่ได้ระบุ"}


ข้อกำหนด:

- สร้างชุดการสอนครบชุด
- สร้างเนื้อหาที่ครูใช้สอนจริง
- สร้างตัวอย่างประกอบ
- สร้างใบงานประมาณ 10-20 ข้อ
- สร้างแบบทดสอบจำนวน {req.question_count} ข้อ
- ใช้เฉพาะประเภทข้อสอบที่เลือก
- ห้ามสร้างประเภทที่ไม่ได้เลือก
- ตรวจสอบคำตอบทุกข้อ
- ตรวจสอบความสอดคล้องของคำถามและเฉลย
- ตัวเลขในการคำนวณต้องถูกต้อง

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


        output_text = response.output_text


        if not output_text:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        data = json.loads(
            output_text
        )


        # เก็บชื่อครูไว้ในข้อมูล
        data["_teacher_name"] = (
            req.teacher_name.strip()
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="AI ส่งข้อมูลกลับมาไม่ใช่ JSON ที่ถูกต้อง"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "สร้างชุดการสอนไม่สำเร็จ: "
                + str(e)
            )
        )


# =========================================================
# PDF STYLES
# =========================================================

def create_styles(font_name):

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "ThaiTitle",

        parent=styles["Title"],

        fontName=font_name,

        fontSize=20,

        leading=25,

        alignment=TA_CENTER,

        spaceAfter=8
    )


    subtitle_style = ParagraphStyle(

        "ThaiSubtitle",

        parent=styles["BodyText"],

        fontName=font_name,

        fontSize=13,

        leading=18,

        alignment=TA_CENTER,

        spaceAfter=10
    )


    h1 = ParagraphStyle(

        "ThaiH1",

        parent=styles["Heading1"],

        fontName=font_name,

        fontSize=15,

        leading=20,

        spaceBefore=10,

        spaceAfter=7
    )


    h2 = ParagraphStyle(

        "ThaiH2",

        parent=styles["Heading2"],

        fontName=font_name,

        fontSize=13,

        leading=18,

        spaceBefore=8,

        spaceAfter=5
    )


    body = ParagraphStyle(

        "ThaiBody",

        parent=styles["BodyText"],

        fontName=font_name,

        fontSize=12,

        leading=19,

        alignment=TA_LEFT,

        spaceAfter=6
    )


    small = ParagraphStyle(

        "ThaiSmall",

        parent=body,

        fontSize=10,

        leading=15
    )


    question = ParagraphStyle(

        "ThaiQuestion",

        parent=body,

        fontName=font_name,

        fontSize=12,

        leading=19,

        spaceAfter=4
    )


    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h1": h1,
        "h2": h2,
        "body": body,
        "small": small,
        "question": question
    }


# =========================================================
# PDF HEADER TABLE
# =========================================================

def build_info_table(
    summary,
    teacher_name,
    styles
):

    teacher = teacher_name.strip()

    if not teacher:

        teacher = "________________________"


    rows = [

        [

            Paragraph(
                "<b>วิชา</b><br/>"
                + esc(summary["subject"]),
                styles["small"]
            ),

            Paragraph(
                "<b>ระดับชั้น</b><br/>"
                + esc(summary["grade"]),
                styles["small"]
            )
        ],

        [

            Paragraph(
                "<b>เรื่อง</b><br/>"
                + esc(summary["topic"]),
                styles["small"]
            ),

            Paragraph(
                "<b>เวลา</b><br/>"
                + esc(summary["duration"]),
                styles["small"]
            )
        ],

        [

            Paragraph(
                "<b>ครูผู้สอน</b><br/>"
                + esc(teacher),
                styles["small"]
            ),

            Paragraph(
                "<b>วันที่</b><br/>"
                "________________________",
                styles["small"]
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
                colors.HexColor("#cfcfcf")
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
                7
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
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


    return table


# =========================================================
# BUILD PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    font_name = register_font()

    styles = create_styles(
        font_name
    )


    buffer = BytesIO()


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=17 * mm,

        bottomMargin=18 * mm,

        title="Teacher Pack"
    )


    story = []


    summary = data["summary"]

    teacher_name = data.get(
        "_teacher_name",
        ""
    )


    # =====================================================
    # HEADER
    # =====================================================

    story.append(

        Paragraph(
            "แผนการจัดการเรียนรู้",
            styles["title"]
        )
    )


    story.append(

        Paragraph(
            "เรื่อง " +
            esc(summary["topic"]),
            styles["subtitle"]
        )
    )


    story.append(

        build_info_table(
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

        lesson = data["lesson_plan"]

        content = data["teaching_content"]


        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                styles["h1"]
            )
        )


        for i, item in enumerate(
            lesson["objective"],
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
                esc(content["intro"]),
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
            content["concepts"],
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


        for i, example in enumerate(
            content["examples"],
            start=1
        ):

            story.append(

                KeepTogether([

                    Paragraph(
                        f"{i}. "
                        f"{esc(example['title'])}",
                        styles["question"]
                    ),

                    Paragraph(
                        esc(
                            example["explanation"]
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
            content["thinking_questions"],
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
                    styles["small"]
                ),

                Paragraph(
                    "<b>กิจกรรม</b>",
                    styles["small"]
                ),

                Paragraph(
                    "<b>รายละเอียด</b>",
                    styles["small"]
                )
            ]
        ]


        for step in lesson["steps"]:

            rows.append(

                [

                    Paragraph(
                        esc(step["time"]),
                        styles["small"]
                    ),

                    Paragraph(
                        esc(step["title"]),
                        styles["small"]
                    ),

                    Paragraph(
                        esc(step["detail"]),
                        styles["small"]
                    )
                ]
            )


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
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#eeeaff")
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#cfcfcf")
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
                    5
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )


        story.append(table)


        story.append(

            Paragraph(
                "4. เคล็ดลับสำหรับครู",
                styles["h1"]
            )
        )


        for i, item in enumerate(
            content["teacher_tips"],
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
                    lesson["assessment"]
                ),
                styles["body"]
            )
        )


    # =====================================================
    # WORKSHEET
    # =====================================================

    def add_worksheet():

        story.append(PageBreak())


        story.append(

            Paragraph(
                "ใบงาน",
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                "เรื่อง " +
                esc(summary["topic"]),
                styles["subtitle"]
            )
        )


        story.append(

            build_info_table(
                summary,
                teacher_name,
                styles
            )
        )


        story.append(
            Spacer(
                1,
                5 * mm
            )
        )


        story.append(

            Paragraph(
                "ชื่อ-สกุล "
                "............................................................",
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
                "ให้นักเรียนอ่านคำถามแต่ละข้อ "
                "และเขียนคำตอบลงในพื้นที่ที่กำหนด",
                styles["body"]
            )
        )


        for item in data["worksheet"]:

            block = [

                Paragraph(
                    f"ข้อ {item['no']}. "
                    f"{esc(item['question'])}",
                    styles["question"]
                ),

                Spacer(
                    1,
                    2 * mm
                ),

                Paragraph(
                    "คำตอบ "
                    "................................................................................",
                    styles["body"]
                ),

                Spacer(
                    1,
                    5 * mm
                )
            ]


            story.append(

                KeepTogether(block)
            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(PageBreak())


        story.append(

            Paragraph(
                "แบบทดสอบ",
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                "เรื่อง " +
                esc(summary["topic"]),
                styles["subtitle"]
            )
        )


        story.append(

            build_info_table(
                summary,
                teacher_name,
                styles
            )
        )


        story.append(
            Spacer(
                1,
                5 * mm
            )
        )


        story.append(

            Paragraph(
                "ชื่อ-สกุล "
                "............................................................",
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
                "ทำแบบทดสอบทุกข้อ "
                "และเลือกหรือเขียนคำตอบให้ถูกต้อง",
                styles["body"]
            )
        )


        for item in data["quiz"]:

            block = []


            block.append(

                Paragraph(
                    f"ข้อ {item['no']} "
                    f"({esc(TYPE_NAMES.get(item['type'], item['type']))})",
                    styles["question"]
                )
            )


            block.append(

                Paragraph(
                    esc(item["question"]),
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

                    Spacer(
                        1,
                        12 * mm
                    )
                )


                block.append(

                    Paragraph(
                        "คำตอบ "
                        "................................................",
                        styles["body"]
                    )
                )


            block.append(

                Spacer(
                    1,
                    4 * mm
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

        story.append(PageBreak())


        story.append(

            Paragraph(
                "เฉลย",
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                "เรื่อง " +
                esc(summary["topic"]),
                styles["subtitle"]
            )
        )


        story.append(

            build_info_table(
                summary,
                teacher_name,
                styles
            )
        )


        story.append(
            Spacer(
                1,
                6 * mm
            )
        )


        story.append(

            Paragraph(
                "เฉลยใบงาน",
                styles["h1"]
            )
        )


        for item in data["worksheet"]:

            story.append(

                Paragraph(
                    f"ข้อ {item['no']}. "
                    f"{esc(item['answer'])}",
                    styles["body"]
                )
            )


        story.append(

            Paragraph(
                "เฉลยแบบทดสอบ",
                styles["h1"]
            )
        )


        for item in data["quiz"]:

            block = [

                Paragraph(
                    f"ข้อ {item['no']}. "
                    f"{esc(item['answer'])}",
                    styles["question"]
                ),

                Paragraph(
                    esc(item["explanation"]),
                    styles["small"]
                ),

                Spacer(
                    1,
                    3 * mm
                )
            ]


            story.append(

                KeepTogether(
                    block
                )
            )


    # =====================================================
    # SELECT SECTION
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
    # FOOTER
    # =====================================================

    def footer(
        canvas,
        doc
    ):

        canvas.saveState()


        # ใช้ Angsana New
        canvas.setFont(
            font_name,
            10
        )


        canvas.setFillColor(
            colors.HexColor("#666666")
        )


        page_text = (
            f"หน้า {doc.page}"
        )


        canvas.drawCentredString(

            A4[0] / 2,

            9 * mm,

            page_text
        )


        canvas.restoreState()


    # =====================================================
    # BUILD
    # =====================================================

    doc.build(

        story,

        onFirstPage=footer,

        onLaterPages=footer
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
