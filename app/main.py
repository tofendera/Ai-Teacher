import os
import json
import html
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
    KeepTogether,
    HRFlowable,
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

FONT_FILE = APP_DIR / "NotoSansThai-Regular.ttf"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.3.0"
)


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
# CONFIG
# =========================================================

MODEL = "gpt-5-mini"


TYPE_NAMES = {
    "multiple_choice": "ปรนัย",
    "fill_blank": "เติมคำ",
    "calculation": "คำนวณ",
    "application": "ประยุกต์ใช้"
}


ALLOWED_TYPES = set(TYPE_NAMES.keys())


# =========================================================
# REQUEST
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

    difficulty: str = "mixed"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """

คุณคือผู้ช่วยจัดทำเอกสารการสอนสำหรับครูไทย

หน้าที่คือเปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็นชุดเอกสารการสอนที่สามารถนำไปใช้จริงได้


ชุดเอกสารประกอบด้วย:

1. ข้อมูลบทเรียน
2. จุดประสงค์การเรียนรู้
3. เนื้อหาที่ครูใช้สอน
4. ตัวอย่างสำหรับอธิบาย
5. คำถามชวนคิด
6. ขั้นตอนการสอน
7. เคล็ดลับสำหรับครู
8. ใบงาน
9. เฉลยใบงาน
10. แบบทดสอบ
11. เฉลยแบบทดสอบ


หลักสำคัญ:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้นจากคำสั่ง
- วิเคราะห์วิชา
- วิเคราะห์หัวข้อ
- วิเคราะห์เวลาเรียน
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น
- ใช้ภาษาที่ครูสามารถนำไปอธิบายกับนักเรียนได้
- หากข้อมูลไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะถ้าไม่แน่ใจ
- ตรวจสอบความถูกต้องก่อนส่ง


เนื้อหาที่ใช้สอน:

ต้องมีเนื้อหาจริง ไม่ใช่เพียงชื่อหัวข้อ

ควรประกอบด้วย:

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบายให้นักเรียนเข้าใจ
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ใบงาน:

ใบงานต้องออกแบบเพื่อให้นักเรียนทำบนกระดาษ

- คำถามชัดเจน
- เหมาะกับระดับชั้น
- สอดคล้องกับเนื้อหาที่สอน
- มีคำตอบที่ตรวจสอบได้
- ไม่จำเป็นต้องเหมือนกับแบบทดสอบ
- หากเป็นคณิตศาสตร์ให้มีโจทย์คำนวณ
- หากเป็นภาษาให้มีแบบฝึกที่เหมาะสม
- หากเป็นวิทยาศาสตร์ให้มีคำถามความเข้าใจและการประยุกต์
- ไม่สร้างโจทย์ซ้ำโดยไม่จำเป็น


แบบทดสอบ:

ต้องสร้างเฉพาะประเภทที่ผู้ใช้เลือก

ประเภท:

multiple_choice
fill_blank
calculation
application


ปรนัย:

- 4 ตัวเลือก
- มีคำตอบถูกเพียงหนึ่งข้อ


เติมคำ:

- มีคำตอบที่ชัดเจน


คำนวณ:

- ตรวจตัวเลข
- ตรวจหน่วย
- ตรวจคำตอบ


ประยุกต์ใช้:

- เป็นสถานการณ์ที่เหมาะกับวัย
- ต้องสัมพันธ์กับบทเรียน


ห้ามสร้างประเภทที่ผู้ใช้ไม่ได้เลือก


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

        "version": "1.3.0",

        "model": MODEL,

        "font_exists": FONT_FILE.exists(),

        "font_path": str(FONT_FILE)
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Render"
        )


    if not req.question_types:

        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ"
        )


    invalid = [

        x

        for x in req.question_types

        if x not in ALLOWED_TYPES
    ]


    if invalid:

        raise HTTPException(

            status_code=400,

            detail=(
                "รูปแบบข้อสอบไม่ถูกต้อง: "
                + ", ".join(invalid)
            )
        )


    selected_types = ", ".join(

        TYPE_NAMES[x]

        for x in req.question_types
    )


    teacher_name = (
        req.teacher_name.strip()
        if req.teacher_name
        else ""
    )


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


ชื่อครูผู้สอน:

{teacher_name if teacher_name else "ไม่ได้ระบุ"}


จำนวนข้อสอบ:

{req.question_count} ข้อ


รูปแบบข้อสอบที่เลือก:

{selected_types}


ระดับความยาก:

{req.difficulty}


สร้าง Teacher Pack ให้ครบถ้วน

ข้อกำหนด:

- สร้างเนื้อหาที่ครูสามารถนำไปสอนได้จริง
- สร้างตัวอย่างประกอบการสอน
- สร้างคำถามชวนคิด
- สร้างใบงานประมาณ 10-20 ข้อ
- สร้างแบบทดสอบจำนวน {req.question_count} ข้อ
- ใช้เฉพาะประเภทข้อสอบที่เลือก
- ห้ามสร้างประเภทที่ไม่ได้เลือก
- ตรวจสอบคำตอบทุกข้อ
- ตรวจสอบตัวเลขและหน่วย
- ตรวจสอบความสัมพันธ์ระหว่างบทเรียน ใบงาน และแบบทดสอบ
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


        output = response.output_text


        if not output:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        data = json.loads(output)


        # เก็บชื่อครูไว้ในข้อมูลที่ส่งกลับ
        data["teacher_name"] = teacher_name


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail="AI ส่งข้อมูลกลับมาไม่ใช่ JSON"
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
# FONT
# =========================================================

_font_registered = False


def register_thai_font():

    global _font_registered


    if not FONT_FILE.exists():

        raise FileNotFoundError(

            f"ไม่พบไฟล์ Font: {FONT_FILE}"
        )


    if not _font_registered:

        pdfmetrics.registerFont(

            TTFont(
                "Thai",
                str(FONT_FILE)
            )
        )

        _font_registered = True


    return "Thai"


# =========================================================
# ESCAPE
# =========================================================

def esc(text):

    return html.escape(
        str(text or ""),
        quote=False
    )


# =========================================================
# STYLES
# =========================================================

def make_styles(font):

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "TitleThai",

        parent=styles["Title"],

        fontName=font,

        fontSize=20,

        leading=27,

        alignment=TA_CENTER,

        spaceAfter=7
    )


    subtitle = ParagraphStyle(

        "SubtitleThai",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=11,

        leading=17,

        alignment=TA_CENTER,

        spaceAfter=12
    )


    h1 = ParagraphStyle(

        "H1Thai",

        parent=styles["Heading1"],

        fontName=font,

        fontSize=15,

        leading=21,

        spaceBefore=12,

        spaceAfter=8
    )


    h2 = ParagraphStyle(

        "H2Thai",

        parent=styles["Heading2"],

        fontName=font,

        fontSize=12,

        leading=18,

        spaceBefore=9,

        spaceAfter=6
    )


    body = ParagraphStyle(

        "BodyThai",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=10.5,

        leading=18,

        alignment=TA_LEFT,

        spaceAfter=7,

        wordWrap="CJK"
    )


    question = ParagraphStyle(

        "QuestionThai",

        parent=body,

        fontSize=11,

        leading=19,

        spaceBefore=3,

        spaceAfter=6
    )


    small = ParagraphStyle(

        "SmallThai",

        parent=body,

        fontSize=9,

        leading=15,

        spaceAfter=4
    )


    return {

        "title": title,
        "subtitle": subtitle,
        "h1": h1,
        "h2": h2,
        "body": body,
        "question": question,
        "small": small
    }


# =========================================================
# HEADER
# =========================================================

def add_document_header(
    story,
    summary,
    teacher_name,
    styles,
    document_title
):

    story.append(

        Paragraph(
            document_title,
            styles["title"]
        )
    )


    story.append(

        Paragraph(
            f"เรื่อง {esc(summary['topic'])}",
            styles["subtitle"]
        )
    )


    teacher_display = (
        teacher_name
        if teacher_name
        else "____________________________"
    )


    data = [

        [

            Paragraph(
                f"<b>วิชา</b><br/>{esc(summary['subject'])}",
                styles["small"]
            ),

            Paragraph(
                f"<b>ระดับชั้น</b><br/>{esc(summary['grade'])}",
                styles["small"]
            )
        ],

        [

            Paragraph(
                f"<b>เวลา</b><br/>{esc(summary['duration'])}",
                styles["small"]
            ),

            Paragraph(
                f"<b>ครูผู้สอน</b><br/>{esc(teacher_display)}",
                styles["small"]
            )
        ]
    ]


    table = Table(

        data,

        colWidths=[
            86 * mm,
            86 * mm
        ]
    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f6f3ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                colors.HexColor("#d9d2ef")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#e5e0f0")
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


    story.append(table)


    story.append(
        Spacer(1, 5 * mm)
    )


# =========================================================
# STUDENT INFO
# =========================================================

def add_student_info(
    story,
    styles
):

    story.append(

        Paragraph(

            "ชื่อ-สกุล "
            "____________________________________________________________",

            styles["body"]
        )
    )


    story.append(

        Paragraph(

            "ชั้น ____________     "
            "เลขที่ ____________     "
            "วันที่ ____________",

            styles["body"]
        )
    )


    story.append(
        Spacer(1, 2 * mm)
    )


# =========================================================
# SECTION BOX
# =========================================================

def add_section_title(
    story,
    title,
    styles
):

    story.append(

        Paragraph(
            title,
            styles["h2"]
        )
    )


# =========================================================
# ANSWER LINES
# =========================================================

def answer_lines(
    story,
    styles,
    count=2
):

    for _ in range(count):

        story.append(

            Paragraph(

                "________________________________________________________________________________",

                styles["body"]
            )
        )


# =========================================================
# BUILD PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    font = register_thai_font()

    styles = make_styles(font)


    buffer = BytesIO()


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=17 * mm,

        leftMargin=17 * mm,

        topMargin=16 * mm,

        bottomMargin=18 * mm,

        title="เอกสารการสอน"
    )


    story = []


    summary = data["summary"]

    teacher_name = (
        data.get("teacher_name")
        or ""
    )


    # =====================================================
    # LESSON PLAN
    # =====================================================

    def add_lesson():

        add_document_header(

            story,
            summary,
            teacher_name,
            styles,
            "แผนการจัดการเรียนรู้"
        )


        lesson = data["lesson_plan"]

        content = data["teaching_content"]


        add_section_title(
            story,
            "1. จุดประสงค์การเรียนรู้",
            styles
        )


        for i, item in enumerate(
            lesson["objective"],
            1
        ):

            story.append(

                Paragraph(
                    f"{i}. {esc(item)}",
                    styles["body"]
                )
            )


        add_section_title(
            story,
            "2. เนื้อหาที่ใช้สอน",
            styles
        )


        story.append(

            Paragraph(
                esc(content["intro"]),
                styles["body"]
            )
        )


        story.append(

            Paragraph(
                "สาระสำคัญ",
                styles["h2"]
            )
        )


        for item in content["concepts"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    styles["body"]
                )
            )


        story.append(

            Paragraph(
                "ตัวอย่างสำหรับใช้สอน",
                styles["h2"]
            )
        )


        for i, example in enumerate(
            content["examples"],
            1
        ):

            story.append(

                KeepTogether([

                    Paragraph(
                        f"{i}. {esc(example['title'])}",
                        styles["question"]
                    ),

                    Paragraph(
                        esc(example["explanation"]),
                        styles["body"]
                    )
                ])
            )


        story.append(

            Paragraph(
                "คำถามชวนคิด",
                styles["h2"]
            )
        )


        for i, item in enumerate(
            content["thinking_questions"],
            1
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
                styles["h2"]
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

            rows.append([

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
            ])


        table = Table(

            rows,

            colWidths=[
                25 * mm,
                42 * mm,
                105 * mm
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
                    0.5,
                    colors.HexColor("#d8d2e8")
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
                    6
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
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


        story.append(table)


        story.append(

            Paragraph(
                "4. เคล็ดลับสำหรับครู",
                styles["h2"]
            )
        )


        for item in content["teacher_tips"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    styles["body"]
                )
            )


        story.append(

            Paragraph(
                "5. การประเมินผล",
                styles["h2"]
            )
        )


        story.append(

            Paragraph(
                esc(lesson["assessment"]),
                styles["body"]
            )
        )


    # =====================================================
    # WORKSHEET
    # =====================================================

    def add_worksheet():

        story.append(PageBreak())


        add_document_header(

            story,
            summary,
            teacher_name,
            styles,
            "ใบงาน"
        )


        add_student_info(
            story,
            styles
        )


        story.append(

            Paragraph(
                "<b>คำชี้แจง</b>",
                styles["h2"]
            )
        )


        story.append(

            Paragraph(

                "ให้นักเรียนอ่านคำถามแต่ละข้อ "
                "และแสดงวิธีคิดหรือเขียนคำตอบลงในพื้นที่ที่กำหนด",

                styles["body"]
            )
        )


        story.append(

            HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor("#d8d2e8"),
                spaceBefore=2 * mm,
                spaceAfter=4 * mm
            )
        )


        for item in data["worksheet"]:

            question = str(
                item["question"]
            )


            if len(question) >= 160:

                lines = 5

            elif len(question) >= 100:

                lines = 4

            elif len(question) >= 50:

                lines = 3

            else:

                lines = 2


            block = []


            block.append(

                Paragraph(

                    f"<b>ข้อ {item['no']}</b> "
                    f"{esc(question)}",

                    styles["question"]
                )
            )


            # ถ้าเป็นโจทย์สั้น ให้พื้นที่มากขึ้น
            block.append(

                Paragraph(
                    "คำตอบ",
                    styles["small"]
                )
            )


            for _ in range(lines):

                block.append(

                    Paragraph(

                        "________________________________________________________________________________",

                        styles["body"]
                    )
                )


            block.append(
                Spacer(1, 4 * mm)
            )


            story.append(

                KeepTogether(block)
            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(PageBreak())


        add_document_header(

            story,
            summary,
            teacher_name,
            styles,
            "แบบทดสอบ"
        )


        add_student_info(
            story,
            styles
        )


        story.append(

            Paragraph(
                "<b>คำชี้แจง</b>",
                styles["h2"]
            )
        )


        story.append(

            Paragraph(

                "ให้นักเรียนอ่านโจทย์และเลือกหรือเขียนคำตอบที่ถูกต้องที่สุด",

                styles["body"]
            )
        )


        story.append(

            HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor("#d8d2e8"),
                spaceBefore=2 * mm,
                spaceAfter=4 * mm
            )
        )


        for item in data["quiz"]:

            block = []


            block.append(

                Paragraph(

                    f"<b>ข้อ {item['no']}</b> "
                    f"<font size='9'>"
                    f"({esc(TYPE_NAMES.get(item['type'], item['type']))})"
                    f"</font>",

                    styles["question"]
                )
            )


            block.append(

                Paragraph(

                    esc(item["question"]),

                    styles["body"]
                )
            )


            options = item.get("options") or []


            if options:

                for i, option in enumerate(options):

                    letter = chr(
                        65 + i
                    )


                    block.append(

                        Paragraph(

                            f"[ ] {letter}. "
                            f"{esc(option)}",

                            styles["body"]
                        )
                    )

            else:

                block.append(

                    Paragraph(
                        "คำตอบ",
                        styles["small"]
                    )
                )


                block.append(

                    Paragraph(

                        "____________________________________________________________",

                        styles["body"]
                    )
                )


            block.append(
                Spacer(1, 4 * mm)
            )


            story.append(

                KeepTogether(block)
            )


    # =====================================================
    # ANSWERS
    # =====================================================

    def add_answers():

        story.append(PageBreak())


        add_document_header(

            story,
            summary,
            teacher_name,
            styles,
            "เฉลย"
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

                    f"<b>ข้อ {item['no']}</b> "
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

            story.append(

                KeepTogether([

                    Paragraph(

                        f"<b>ข้อ {item['no']}</b> "
                        f"{esc(item['answer'])}",

                        styles["question"]
                    ),

                    Paragraph(

                        esc(item["explanation"]),

                        styles["small"]
                    )
                ])
            )


    # =====================================================
    # SECTIONS
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
            font,
            8
        )


        canvas.setFillColor(
            colors.HexColor("#777777")
        )


        canvas.drawCentredString(

            A4[0] / 2,

            9 * mm,

            f"หน้า {doc.page}"
        )


        canvas.restoreState()


    doc.build(

        story,

        onFirstPage=footer,

        onLaterPages=footer
    )


    buffer.seek(0)

    return buffer


# =========================================================
# PDF API
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
