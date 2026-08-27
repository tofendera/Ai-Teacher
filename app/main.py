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
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable
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
    title="AI ครูผู้ช่วย",
    version="1.2.0"
)


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
# MODEL
# =========================================================

MODEL = "gpt-5-mini"


# =========================================================
# QUESTION TYPES
# =========================================================

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

เปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็น Teacher Pack พร้อมใช้จริง

ตัวอย่าง:

"เศษส่วน ป.4 1 ชั่วโมง"


Teacher Pack ประกอบด้วย:

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
- วิเคราะห์เวลาเรียน
- เนื้อหาเหมาะกับวัย
- เนื้อหาเหมาะกับระดับชั้น
- ถ้าข้อมูลไม่ชัด ให้ใช้บริบทสมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะถ้าไม่แน่ใจ
- ตรวจสอบความถูกต้องก่อนส่ง


เนื้อหาที่ใช้สอน:

ต้องเขียนให้ครูนำไปพูดหรืออธิบายกับนักเรียนได้จริง

ควรมี:

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบาย
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ใบงาน:

ใบงานเป็นแบบฝึกฝนสำหรับนักเรียน

ต้องสอดคล้องกับบทเรียน

แต่ไม่จำเป็นต้องมีรูปแบบเดียวกับแบบทดสอบ

แต่ละข้อควรมีคำตอบที่ตรวจสอบได้


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

- คำตอบชัดเจน


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

        "app": "AI ครูผู้ช่วย",

        "version": "1.2.0",

        "model": MODEL,

        "font_exists": FONT_FILE.exists(),

        "font_path": str(FONT_FILE)
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


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


จำนวนข้อสอบ:

{req.question_count} ข้อ


ประเภทข้อสอบ:

{selected_types}


ระดับความยาก:

{req.difficulty}


ให้สร้าง Teacher Pack ครบชุด

ข้อกำหนดเพิ่มเติม:

- สร้างเนื้อหาที่ครูใช้สอนได้จริง
- สร้างตัวอย่าง
- สร้างคำถามชวนคิด
- สร้างใบงานประมาณ 10-20 ข้อ
- สร้างแบบทดสอบ {req.question_count} ข้อ
- ใช้เฉพาะประเภทที่เลือก
- ห้ามสร้างประเภทอื่น
- ตรวจคำตอบทุกข้อ
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


        return json.loads(output)


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

    return (

        str(text or "")

        .replace("&", "&amp;")

        .replace("<", "&lt;")

        .replace(">", "&gt;")
    )


# =========================================================
# PDF STYLES
# =========================================================

def make_styles(font):

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "ThaiTitle",

        parent=styles["Title"],

        fontName=font,

        fontSize=21,

        leading=28,

        alignment=TA_CENTER,

        spaceAfter=8
    )


    subtitle = ParagraphStyle(

        "ThaiSubtitle",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=11,

        leading=17,

        alignment=TA_CENTER,

        spaceAfter=12
    )


    h1 = ParagraphStyle(

        "ThaiH1",

        parent=styles["Heading1"],

        fontName=font,

        fontSize=15,

        leading=21,

        spaceBefore=12,

        spaceAfter=8
    )


    h2 = ParagraphStyle(

        "ThaiH2",

        parent=styles["Heading2"],

        fontName=font,

        fontSize=12,

        leading=18,

        spaceBefore=9,

        spaceAfter=6
    )


    body = ParagraphStyle(

        "ThaiBody",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=10.5,

        leading=18,

        alignment=TA_LEFT,

        spaceAfter=7,

        wordWrap="CJK"
    )


    question = ParagraphStyle(

        "ThaiQuestion",

        parent=body,

        fontSize=11,

        leading=19,

        spaceBefore=5,

        spaceAfter=5
    )


    small = ParagraphStyle(

        "ThaiSmall",

        parent=body,

        fontSize=9,

        leading=15,

        spaceAfter=4
    )


    answer_line = ParagraphStyle(

        "ThaiAnswer",

        parent=body,

        fontSize=10,

        leading=18,

        spaceBefore=3,

        spaceAfter=8
    )


    return {

        "title": title,

        "subtitle": subtitle,

        "h1": h1,

        "h2": h2,

        "body": body,

        "question": question,

        "small": small,

        "answer": answer_line
    }


# =========================================================
# COMMON HEADER
# =========================================================

def document_header(
    story,
    summary,
    styles
):

    story.append(

        Paragraph(
            "AI ครูผู้ช่วย",
            styles["title"]
        )
    )


    story.append(

        Paragraph(
            "ชุดการสอนพร้อมใช้",
            styles["subtitle"]
        )
    )


    info_data = [

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
                f"<b>เรื่อง</b><br/>{esc(summary['topic'])}",
                styles["small"]
            ),

            Paragraph(
                f"<b>เวลา</b><br/>{esc(summary['duration'])}",
                styles["small"]
            )
        ]
    ]


    table = Table(

        info_data,

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
                colors.HexColor("#f5f2ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#ddd7f5")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#e4e0f2")
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
        Spacer(1, 6 * mm)
    )


# =========================================================
# STUDENT INFO
# =========================================================

def student_info(
    story,
    styles
):

    story.append(

        Paragraph(

            "ชื่อ-สกุล "
            "______________________________________________",

            styles["body"]
        )
    )


    story.append(

        Paragraph(

            "ชั้น ____________ "
            "เลขที่ ____________ "
            "วันที่ ____________",

            styles["body"]
        )
    )


    story.append(
        Spacer(1, 3 * mm)
    )


# =========================================================
# ANSWER SPACE
# =========================================================

def answer_space(
    story,
    styles,
    lines=2
):

    for _ in range(lines):

        story.append(

            Paragraph(

                "________________________________________________________________________________",

                styles["answer"]
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

        bottomMargin=17 * mm,

        title="AI ครูผู้ช่วย",

        author="AI ครูผู้ช่วย"
    )


    story = []

    summary = data["summary"]


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        document_header(
            story,
            summary,
            styles
        )


        lesson = data["lesson_plan"]

        content = data["teaching_content"]


        story.append(

            Paragraph(
                "แผนการสอน",
                styles["h1"]
            )
        )


        story.append(

            Paragraph(
                "จุดประสงค์การเรียนรู้",
                styles["h2"]
            )
        )


        for item in lesson["objective"]:

            story.append(

                Paragraph(
                    "• " + esc(item),
                    styles["body"]
                )
            )


        story.append(

            Paragraph(
                "เนื้อหาที่ใช้สอน",
                styles["h2"]
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


        for example in content["examples"]:

            story.append(

                KeepTogether([

                    Paragraph(
                        esc(example["title"]),
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
                "ขั้นตอนการจัดการเรียนรู้",
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
                "เคล็ดลับสำหรับครู",
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
                "การประเมินผล",
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


        story.append(

            Paragraph(
                "ใบงาน",
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                f"เรื่อง {esc(summary['topic'])}",
                styles["subtitle"]
            )
        )


        student_info(
            story,
            styles
        )


        story.append(

            Paragraph(
                "<b>คำชี้แจง</b> "
                "ให้นักเรียนอ่านโจทย์และตอบคำถามให้ครบทุกข้อ",
                styles["body"]
            )
        )


        story.append(

            HRFlowable(
                width="100%",
                thickness=0.7,
                color=colors.HexColor("#ddd7e8"),
                spaceBefore=3 * mm,
                spaceAfter=3 * mm
            )
        )


        for item in data["worksheet"]:

            block = []


            block.append(

                Paragraph(

                    f"<b>{item['no']}.</b> "
                    f"{esc(item['question'])}",

                    styles["question"]
                )
            )


            # เพิ่มพื้นที่ตอบตามลักษณะโจทย์
            question_text = str(
                item["question"]
            )


            if len(question_text) > 140:

                lines = 4

            elif len(question_text) > 80:

                lines = 3

            else:

                lines = 2


            for _ in range(lines):

                block.append(

                    Paragraph(

                        "________________________________________________________________________________",

                        styles["answer"]
                    )
                )


            story.append(

                KeepTogether(block)
            )


            story.append(
                Spacer(1, 2 * mm)
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
                f"เรื่อง {esc(summary['topic'])}",
                styles["subtitle"]
            )
        )


        student_info(
            story,
            styles
        )


        story.append(

            Paragraph(

                "<b>คำชี้แจง:</b> "
                "ให้นักเรียนอ่านโจทย์และเลือกหรือเขียนคำตอบที่ถูกต้อง",

                styles["body"]
            )
        )


        story.append(

            HRFlowable(
                width="100%",
                thickness=0.7,
                color=colors.HexColor("#ddd7e8"),
                spaceBefore=3 * mm,
                spaceAfter=4 * mm
            )
        )


        for item in data["quiz"]:

            block = []


            block.append(

                Paragraph(

                    f"<b>ข้อ {item['no']}</b> "
                    f"<font size='9'>"
                    f"[{esc(TYPE_NAMES.get(item['type'], item['type']))}"
                    f"]</font>",

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

                            f"☐ {letter}. "
                            f"{esc(option)}",

                            styles["body"]
                        )
                    )

            else:

                block.append(

                    Spacer(
                        1,
                        2 * mm
                    )
                )


                block.append(

                    Paragraph(

                        "คำตอบ: "
                        "____________________________________________",

                        styles["answer"]
                    )
                )


            block.append(
                Spacer(
                    1,
                    3 * mm
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
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                f"เรื่อง {esc(summary['topic'])}",
                styles["subtitle"]
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

                    f"<b>ข้อ {item['no']}.</b> "
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

                        f"<b>ข้อ {item['no']}.</b> "
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

            f"AI ครูผู้ช่วย  •  หน้า {doc.page}"
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
