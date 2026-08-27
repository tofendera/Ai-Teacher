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
from reportlab.lib.enums import TA_CENTER
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

# main.py อยู่ใน:
# Ai-Teacher/app/main.py
#
# ดังนั้น parent.parent = Ai-Teacher

PROJECT_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = PROJECT_DIR / "static"

INDEX_FILE = STATIC_DIR / "index.html"

FONT_DIR = PROJECT_DIR / "fonts"

THAI_FONT_FILE = FONT_DIR / "NotoSansThai-Regular.ttf"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI ครูผู้ช่วย",
    version="1.2.0"
)


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
# OPENAI
# =========================================================

MODEL = "gpt-5-mini"


# =========================================================
# QUESTION TYPE NAMES
# =========================================================

# ต้องประกาศด้านนอก generate()
# เพราะ build_pdf() ก็ต้องใช้งานด้วย

TYPE_NAMES = {

    "multiple_choice": "ปรนัย",

    "fill_blank": "เติมคำ",

    "calculation": "คำนวณ",

    "application": "ประยุกต์ใช้"
}


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

    difficulty: str = "mixed"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """

คุณคือ AI ครูผู้ช่วยสำหรับครูไทย

หน้าที่ของคุณคือเปลี่ยนคำสั่งสั้น ๆ
ของครู เช่น

"เศษส่วน ป.4 1 ชั่วโมง"

ให้กลายเป็น Teacher Pack พร้อมใช้


Teacher Pack ประกอบด้วย:

1. ข้อมูลสรุปบทเรียน
2. จุดประสงค์การเรียนรู้
3. เนื้อหาที่ครูใช้สอน
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
- วิเคราะห์เวลาเรียน
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น

- หากข้อมูลบางส่วนไม่ชัด
  ให้ใช้บริบทที่สมเหตุสมผล

- ห้ามอ้างหลักสูตรเฉพาะ
  ถ้าไม่แน่ใจ

- ห้ามสร้างข้อมูลมั่ว

- ตรวจสอบความถูกต้องก่อนส่ง


ส่วนเนื้อหาที่ใช้สอน:

ต้องเขียนให้ครูสามารถนำไปอธิบาย
กับนักเรียนได้จริง

ควรประกอบด้วย:

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบาย
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ส่วนใบงาน:

ใบงานคือแบบฝึกฝน
สำหรับให้นักเรียนฝึกทักษะ

ไม่จำเป็นต้องมีรูปแบบเดียวกับแบบทดสอบ

แต่ต้องสอดคล้องกับหัวข้อ


ส่วนแบบทดสอบ:

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

- มีคำตอบชัดเจน


คำนวณ:

- ตรวจตัวเลข
- ตรวจหน่วย
- ตรวจคำตอบ


ประยุกต์ใช้:

- เป็นสถานการณ์ที่เหมาะกับวัย
- ต้องสัมพันธ์กับเนื้อหา


ห้ามสร้างประเภทข้อสอบ
ที่ผู้ใช้ไม่ได้เลือก


ตอบเป็น JSON ตาม Schema เท่านั้น

"""


# =========================================================
# JSON SCHEMA
# =========================================================

SCHEMA = {

    "type": "object",

    "additionalProperties": False,

    "properties": {

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

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


        # -------------------------------------------------
        # LESSON PLAN
        # -------------------------------------------------

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


        # -------------------------------------------------
        # TEACHING CONTENT
        # -------------------------------------------------

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


        # -------------------------------------------------
        # WORKSHEET
        # -------------------------------------------------

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


        # -------------------------------------------------
        # QUIZ
        # -------------------------------------------------

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
# HOME PAGE
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
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "app": "AI ครูผู้ช่วย",

        "version": "1.2.0",

        "model": MODEL
    }


# =========================================================
# GENERATE TEACHER PACK
# =========================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise HTTPException(
            status_code=500,
            detail=(
                "ยังไม่ได้ตั้งค่า "
                "OPENAI_API_KEY "
                "ใน Render"
            )
        )


    # -----------------------------------------------------
    # ALLOWED TYPES
    # -----------------------------------------------------

    allowed_types = {

        "multiple_choice",

        "fill_blank",

        "calculation",

        "application"
    }


    # -----------------------------------------------------
    # CHECK TYPES
    # -----------------------------------------------------

    if not req.question_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "กรุณาเลือกรูปแบบข้อสอบ "
                "อย่างน้อย 1 แบบ"
            )
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


    # -----------------------------------------------------
    # TYPE NAMES
    # -----------------------------------------------------

    selected_types = ", ".join(

        TYPE_NAMES[t]

        for t in req.question_types
    )


    # -----------------------------------------------------
    # USER PROMPT
    # -----------------------------------------------------

    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


จำนวนข้อสอบ:

{req.question_count} ข้อ


รูปแบบข้อสอบที่เลือก:

{selected_types}


ระดับความยาก:

{req.difficulty}


ข้อกำหนดเพิ่มเติม:

- สร้าง Teacher Pack ครบชุด
- สร้างเนื้อหาที่ครูใช้สอนจริง
- สร้างตัวอย่างประกอบการสอน
- สร้างใบงานประมาณ 10-20 ข้อ
- สร้างแบบทดสอบจำนวน {req.question_count} ข้อ
- แบบทดสอบต้องใช้เฉพาะประเภทที่เลือก
- ห้ามสร้างประเภทที่ไม่ได้เลือก
- ตรวจสอบเฉลยทุกข้อ
- ตรวจสอบความสอดคล้องระหว่างคำถามและคำตอบ
"""


    # -----------------------------------------------------
    # OPENAI
    # -----------------------------------------------------

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


        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

        output_text = response.output_text


        if not output_text:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        data = json.loads(
            output_text
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail=(
                "AI ส่งข้อมูลกลับมา "
                "ไม่ใช่ JSON ที่ถูกต้อง"
            )
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
# THAI FONT
# =========================================================

def register_thai_font():

    """
    โหลดฟอนต์ภาษาไทยจากโปรเจกต์โดยตรง

    โครงสร้างที่ต้องมี:

    Ai-Teacher/
        app/
            main.py
        fonts/
            NotoSansThai-Regular.ttf
    """

    # -----------------------------------------------------
    # CHECK FONT DIRECTORY
    # -----------------------------------------------------

    if not FONT_DIR.exists():

        raise RuntimeError(
            "ไม่พบโฟลเดอร์ fonts: "
            + str(FONT_DIR)
        )


    # -----------------------------------------------------
    # CHECK FONT FILE
    # -----------------------------------------------------

    if not THAI_FONT_FILE.exists():

        raise RuntimeError(
            "ไม่พบไฟล์ฟอนต์ภาษาไทย: "
            + str(THAI_FONT_FILE)
        )


    # -----------------------------------------------------
    # REGISTER FONT
    # -----------------------------------------------------

    try:

        # ป้องกัน register ซ้ำ
        if "Thai" not in pdfmetrics.getRegisteredFontNames():

            pdfmetrics.registerFont(
                TTFont(
                    "Thai",
                    str(THAI_FONT_FILE)
                )
            )


        return "Thai"


    except Exception as e:

        raise RuntimeError(
            "โหลดฟอนต์ภาษาไทยไม่สำเร็จ: "
            + str(e)
        )


# =========================================================
# ESCAPE HTML
# =========================================================

def esc(text):

    return (

        str(text or "")

        .replace("&", "&amp;")

        .replace("<", "&lt;")

        .replace(">", "&gt;")

    )


# =========================================================
# BUILD PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    # -----------------------------------------------------
    # REGISTER THAI FONT
    # -----------------------------------------------------

    thai_font = register_thai_font()


    # -----------------------------------------------------
    # BUFFER
    # -----------------------------------------------------

    buffer = BytesIO()


    # -----------------------------------------------------
    # DOCUMENT
    # -----------------------------------------------------

    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=16 * mm,

        leftMargin=16 * mm,

        topMargin=16 * mm,

        bottomMargin=16 * mm,

        title="AI ครูผู้ช่วย"
    )


    # -----------------------------------------------------
    # STYLES
    # -----------------------------------------------------

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "TitleThai",

        parent=styles["Title"],

        fontName=thai_font,

        fontSize=20,

        leading=26,

        alignment=TA_CENTER,

        spaceAfter=8
    )


    h1 = ParagraphStyle(

        "H1Thai",

        parent=styles["Heading1"],

        fontName=thai_font,

        fontSize=15,

        leading=21,

        spaceBefore=10,

        spaceAfter=7
    )


    h2 = ParagraphStyle(

        "H2Thai",

        parent=styles["Heading2"],

        fontName=thai_font,

        fontSize=12,

        leading=18,

        spaceBefore=8,

        spaceAfter=5
    )


    body = ParagraphStyle(

        "BodyThai",

        parent=styles["BodyText"],

        fontName=thai_font,

        fontSize=10.5,

        leading=18,

        spaceAfter=5
    )


    small = ParagraphStyle(

        "SmallThai",

        parent=body,

        fontName=thai_font,

        fontSize=9,

        leading=14
    )


    question_style = ParagraphStyle(

        "QuestionThai",

        parent=body,

        fontName=thai_font,

        spaceAfter=3
    )


    # -----------------------------------------------------
    # STORY
    # -----------------------------------------------------

    story = []


    summary = data.get(
        "summary",
        {}
    )


    # =====================================================
    # HEADER
    # =====================================================

    story.append(

        Paragraph(
            "AI ครูผู้ช่วย",
            title_style
        )

    )


    story.append(

        Paragraph(
            "ชุดการสอนพร้อมใช้",
            h2
        )

    )


    story.append(

        Paragraph(

            f"<b>วิชา:</b> "
            f"{esc(summary.get('subject'))}"
            f"&nbsp;&nbsp;&nbsp;"
            f"<b>ชั้น:</b> "
            f"{esc(summary.get('grade'))}",

            body
        )

    )


    story.append(

        Paragraph(

            f"<b>เรื่อง:</b> "
            f"{esc(summary.get('topic'))}"
            f"&nbsp;&nbsp;&nbsp;"
            f"<b>เวลา:</b> "
            f"{esc(summary.get('duration'))}",

            body
        )

    )


    story.append(

        Spacer(
            1,
            5 * mm
        )

    )


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        lesson = data["lesson_plan"]

        content = data["teaching_content"]


        # -------------------------------------------------
        # OBJECTIVE
        # -------------------------------------------------

        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                h1
            )

        )


        for item in lesson["objective"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    body
                )

            )


        # -------------------------------------------------
        # TEACHING CONTENT
        # -------------------------------------------------

        story.append(

            Paragraph(
                "2. เนื้อหาที่ใช้สอน",
                h1
            )

        )


        story.append(

            Paragraph(
                esc(content["intro"]),
                body
            )

        )


        # -------------------------------------------------
        # CONCEPTS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "สาระสำคัญ",
                h2
            )

        )


        for item in content["concepts"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    body
                )

            )


        # -------------------------------------------------
        # EXAMPLES
        # -------------------------------------------------

        story.append(

            Paragraph(
                "ตัวอย่างสำหรับใช้สอน",
                h2
            )

        )


        for example in content["examples"]:

            story.append(

                KeepTogether([

                    Paragraph(

                        esc(
                            example["title"]
                        ),

                        question_style
                    ),

                    Paragraph(

                        esc(
                            example["explanation"]
                        ),

                        body
                    )

                ])

            )


        # -------------------------------------------------
        # THINKING QUESTIONS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "คำถามชวนคิด",
                h2
            )

        )


        for item in content["thinking_questions"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    body
                )

            )


        # -------------------------------------------------
        # LESSON STEPS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "3. ขั้นตอนการจัดการเรียนรู้",
                h1
            )

        )


        rows = [

            [

                Paragraph(
                    "<b>เวลา</b>",
                    small
                ),

                Paragraph(
                    "<b>กิจกรรม</b>",
                    small
                ),

                Paragraph(
                    "<b>รายละเอียด</b>",
                    small
                )

            ]

        ]


        for step in lesson["steps"]:

            rows.append(

                [

                    Paragraph(
                        esc(step["time"]),
                        small
                    ),

                    Paragraph(
                        esc(step["title"]),
                        small
                    ),

                    Paragraph(
                        esc(step["detail"]),
                        small
                    )

                ]

            )


        table = Table(

            rows,

            colWidths=[

                24 * mm,

                40 * mm,

                112 * mm

            ],

            repeatRows=1
        )


        table.setStyle(

            TableStyle([

                (

                    "BACKGROUND",

                    (0, 0),

                    (-1, 0),

                    colors.HexColor(
                        "#eeeaff"
                    )

                ),

                (

                    "GRID",

                    (0, 0),

                    (-1, -1),

                    0.4,

                    colors.HexColor(
                        "#d9d4e8"
                    )

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


        # -------------------------------------------------
        # TEACHER TIPS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "4. เคล็ดลับสำหรับครู",
                h1
            )

        )


        for item in content["teacher_tips"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    body
                )

            )


        # -------------------------------------------------
        # ASSESSMENT
        # -------------------------------------------------

        story.append(

            Paragraph(
                "5. การประเมินผล",
                h1
            )

        )


        story.append(

            Paragraph(
                esc(lesson["assessment"]),
                body
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
                title_style
            )

        )


        story.append(

            Paragraph(

                f"<b>เรื่อง:</b> "
                f"{esc(summary.get('topic'))}",

                body
            )

        )


        story.append(

            Paragraph(

                "ชื่อ................................................ "
                "ชั้น........ เลขที่........",

                body
            )

        )


        story.append(

            Spacer(
                1,
                4 * mm
            )

        )


        for item in data["worksheet"]:

            story.append(

                KeepTogether([

                    Paragraph(

                        f"{item['no']}. "
                        f"{esc(item['question'])}",

                        body
                    ),

                    Paragraph(

                        "คำตอบ: "
                        "................................................................................",

                        body
                    ),

                    Spacer(
                        1,
                        2 * mm
                    )

                ])

            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(PageBreak())


        story.append(

            Paragraph(
                "แบบทดสอบ",
                title_style
            )

        )


        story.append(

            Paragraph(

                f"<b>เรื่อง:</b> "
                f"{esc(summary.get('topic'))}",

                body
            )

        )


        story.append(

            Paragraph(

                "ชื่อ................................................ "
                "ชั้น........ เลขที่........",

                body
            )

        )


        story.append(

            Paragraph(

                "คำชี้แจง: ทำแบบทดสอบตามคำสั่งของแต่ละข้อ",

                body
            )

        )


        for item in data["quiz"]:

            block = [

                Paragraph(

                    f"ข้อ {item['no']} "
                    f"[{esc(TYPE_NAMES.get(item['type'], item['type']))}]",

                    question_style
                ),

                Paragraph(

                    esc(item["question"]),

                    body
                )

            ]


            # -------------------------------------------------
            # OPTIONS
            # -------------------------------------------------

            if item.get("options"):

                for i, option in enumerate(
                    item["options"]
                ):

                    letter = chr(
                        65 + i
                    )


                    block.append(

                        Paragraph(

                            f"{letter}. "
                            f"{esc(option)}",

                            body
                        )

                    )


            block.append(

                Spacer(
                    1,
                    2 * mm
                )

            )


            story.append(

                KeepTogether(block)

            )


    # =====================================================
    # ANSWERS
    # =====================================================

    def add_answers():

        story.append(PageBreak())


        story.append(

            Paragraph(
                "เฉลย",
                title_style
            )

        )


        # -------------------------------------------------
        # WORKSHEET ANSWERS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "เฉลยใบงาน",
                h1
            )

        )


        for item in data["worksheet"]:

            story.append(

                Paragraph(

                    f"{item['no']}. "
                    f"{esc(item['answer'])}",

                    body
                )

            )


        # -------------------------------------------------
        # QUIZ ANSWERS
        # -------------------------------------------------

        story.append(

            Paragraph(
                "เฉลยแบบทดสอบ",
                h1
            )

        )


        for item in data["quiz"]:

            story.append(

                KeepTogether([

                    Paragraph(

                        f"ข้อ {item['no']} — "
                        f"{esc(item['answer'])}",

                        question_style
                    ),

                    Paragraph(

                        esc(
                            item["explanation"]
                        ),

                        small
                    )

                ])

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


        canvas.setFont(
            thai_font,
            8
        )


        canvas.setFillColor(
            colors.HexColor(
                "#777777"
            )
        )


        canvas.drawCentredString(

            A4[0] / 2,

            9 * mm,

            f"AI ครูผู้ช่วย • หน้า {doc.page}"

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

    # -----------------------------------------------------
    # ALLOWED SECTIONS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # BUILD PDF
    # -----------------------------------------------------

    try:

        pdf_file = build_pdf(

            data,

            section

        )


        filename = (

            f"ai-teacher-{section}.pdf"

        )


        return StreamingResponse(

            pdf_file,

            media_type="application/pdf",

            headers={

                "Content-Disposition":

                f'attachment; filename="{filename}"'

            }

        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(

                "สร้าง PDF ไม่สำเร็จ: "

                + str(e)

            )

        )
