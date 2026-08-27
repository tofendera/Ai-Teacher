import os
import json
from pathlib import Path
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
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

FONT_DIR = BASE_DIR / "fonts"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.5.0"
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
# OPENAI
# =========================================================

MODEL = "gpt-5-mini"


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

คุณเป็นผู้ช่วยจัดเตรียมเอกสารการเรียนการสอนสำหรับครูไทย

ให้เปลี่ยนคำสั่งสั้น ๆ ของครู เช่น

"ระบบสุริยะ ป.5 1 ชั่วโมง"

ให้กลายเป็นชุดเอกสารการเรียนการสอนที่นำไปใช้ได้จริง


ชุดเอกสารประกอบด้วย:

1. สรุปข้อมูลบทเรียน
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


หลักสำคัญ:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้น วิชา หัวข้อ และเวลา
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น
- ใช้ภาษาที่ครูสามารถนำไปใช้สอนได้จริง
- หลีกเลี่ยงการเขียนแบบสั้นจนเกินไป
- แบ่งเนื้อหาเป็นย่อหน้าอย่างเป็นธรรมชาติ
- ไม่ควรใส่ข้อมูลที่ไม่แน่ใจ
- หากข้อมูลไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ตรวจสอบความถูกต้องของคำตอบทุกข้อ


เนื้อหาการสอน:

ควรประกอบด้วย

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบายให้นักเรียนเข้าใจ
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ใบงาน:

- เหมาะกับระดับชั้น
- มีคำถามที่ชัดเจน
- มีพื้นที่ให้นักเรียนตอบ
- มีความหลากหลายพอสมควร
- ไม่ควรซ้ำกับแบบทดสอบทุกข้อ


แบบทดสอบ:

สร้างเฉพาะประเภทที่ผู้ใช้เลือก

ประเภทที่รองรับ:

multiple_choice
fill_blank
calculation
application


ปรนัย:

- 4 ตัวเลือก
- มีคำตอบถูกเพียงหนึ่งข้อ


เติมคำ:

- คำตอบต้องชัดเจน


คำนวณ:

- ตรวจตัวเลข
- ตรวจหน่วย
- ตรวจคำตอบ


ประยุกต์ใช้:

- เป็นสถานการณ์เหมาะกับวัย
- เชื่อมโยงกับชีวิตประจำวัน
- ต้องสัมพันธ์กับบทเรียน


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

                "object": {

                    "type": "object"
                },

                "type": {

                    "type": "string"
                }
            }
        }
    }
}


# =========================================================
# FIX QUIZ SCHEMA
# =========================================================

SCHEMA["properties"]["quiz"] = {

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


SCHEMA["required"] = [
    "summary",
    "lesson_plan",
    "teaching_content",
    "worksheet",
    "quiz"
]


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

        "model": MODEL
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

        t for t in req.question_types
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


    type_names = {

        "multiple_choice": "ปรนัย",
        "fill_blank": "เติมคำ",
        "calculation": "คำนวณ",
        "application": "ประยุกต์ใช้"
    }


    selected_types = ", ".join(

        type_names[t]
        for t in req.question_types

    )


    teacher_name = (
        req.teacher_name.strip()
        if req.teacher_name
        else "ไม่ระบุ"
    )


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


ชื่อครูผู้สอน:

{teacher_name}


จำนวนข้อสอบ:

{req.question_count} ข้อ


รูปแบบข้อสอบที่เลือก:

{selected_types}


ระดับความยาก:

{req.difficulty}


ข้อกำหนดเพิ่มเติม:

- สร้างชุดเอกสารครบชุด
- ใช้ชื่อครูผู้สอนเป็น "{teacher_name}"
- สร้างเนื้อหาที่ครูสามารถนำไปใช้จริง
- สร้างใบงานประมาณ 10-20 ข้อ
- สร้างแบบทดสอบจำนวน {req.question_count} ข้อ
- ใช้เฉพาะประเภทข้อสอบที่เลือก
- ห้ามสร้างประเภทที่ไม่ได้เลือก
- ตรวจสอบเฉลยทุกข้อ
- ตรวจสอบตัวเลข
- ตรวจสอบความสอดคล้องระหว่างคำถามกับคำตอบ
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


        # ป้องกันข้อมูลเก่าที่ไม่มีชื่อครู

        if "summary" not in data:

            data["summary"] = {}


        data["summary"]["teacher_name"] = teacher_name


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
# FONT
# =========================================================

FONT_REGULAR = None


def register_thai_font():

    global FONT_REGULAR

    if FONT_REGULAR:

        return FONT_REGULAR


    candidates = [

        BASE_DIR / "fonts" / "NotoSansThai-Regular.ttf",

        BASE_DIR / "NotoSansThai-Regular.ttf",

        APP_DIR / "NotoSansThai-Regular.ttf",

        Path("/app/fonts/NotoSansThai-Regular.ttf"),

        Path("/app/NotoSansThai-Regular.ttf"),

        Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),

        Path("/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf")
    ]


    font_file = None


    for path in candidates:

        if path.exists():

            font_file = path
            break


    if not font_file:

        # ค้นหาอัตโนมัติอีกครั้ง

        try:

            matches = list(
                BASE_DIR.rglob(
                    "NotoSansThai-Regular.ttf"
                )
            )

            if matches:

                font_file = matches[0]

        except Exception:

            pass


    if not font_file:

        raise RuntimeError(
            "ไม่พบไฟล์ Font: "
            "NotoSansThai-Regular.ttf"
        )


    font_name = "ThaiRegularV15"


    if font_name not in pdfmetrics.getRegisteredFontNames():

        pdfmetrics.registerFont(

            TTFont(
                font_name,
                str(font_file)
            )
        )


    FONT_REGULAR = font_name

    return FONT_REGULAR


# =========================================================
# ESCAPE
# =========================================================

def esc(text):

    if text is None:

        return ""

    return (

        str(text)

        .replace("&", "&amp;")

        .replace("<", "&lt;")

        .replace(">", "&gt;")

        .replace('"', "&quot;")

        .replace("'", "&#039;")

    )


# =========================================================
# TEXT HELPERS
# =========================================================

def paragraph(
    text,
    style
):

    return Paragraph(
        esc(text).replace(
            "\n",
            "<br/>"
        ),
        style
    )


def bullet_paragraph(
    text,
    style
):

    return Paragraph(

        "• " + esc(text),

        style
    )


# =========================================================
# NUMBERED CANVAS
# =========================================================

from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        canvas.Canvas.__init__(
            self,
            *args,
            **kwargs
        )

        self._saved_page_states = []


    def showPage(self):

        self._saved_page_states.append(
            dict(self.__dict__)
        )

        self._startPage()


    def save(self):

        page_count = len(
            self._saved_page_states
        )


        for state in self._saved_page_states:

            self.__dict__.update(
                state
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

        regular = register_thai_font()


        self.saveState()


        self.setFont(
            regular,
            8
        )


        self.setFillColor(
            colors.HexColor(
                "#666666"
            )
        )


        self.drawCentredString(

            A4[0] / 2,

            9 * mm,

            f"หน้า {self._pageNumber} จาก {page_count}"

        )


        self.restoreState()


# =========================================================
# STYLES
# =========================================================

def create_styles():

    font = register_thai_font()

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "FormalTitle",

        parent=styles["Title"],

        fontName=font,

        fontSize=18,

        leading=24,

        alignment=TA_CENTER,

        spaceAfter=5,

        textColor=colors.HexColor("#222222")
    )


    subtitle = ParagraphStyle(

        "FormalSubtitle",

        parent=styles["Normal"],

        fontName=font,

        fontSize=11,

        leading=17,

        alignment=TA_CENTER,

        spaceAfter=12
    )


    section = ParagraphStyle(

        "FormalSection",

        parent=styles["Heading1"],

        fontName=font,

        fontSize=13,

        leading=20,

        spaceBefore=10,

        spaceAfter=8,

        textColor=colors.HexColor("#222222"),

        keepWithNext=True
    )


    subsection = ParagraphStyle(

        "FormalSubsection",

        parent=styles["Heading2"],

        fontName=font,

        fontSize=11.5,

        leading=18,

        spaceBefore=7,

        spaceAfter=5,

        keepWithNext=True
    )


    body = ParagraphStyle(

        "FormalBody",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=10.5,

        leading=18,

        alignment=TA_JUSTIFY,

        firstLineIndent=8 * mm,

        spaceAfter=7,

        textColor=colors.HexColor("#222222")
    )


    body_no_indent = ParagraphStyle(

        "FormalBodyNoIndent",

        parent=body,

        firstLineIndent=0,

        spaceAfter=6
    )


    bullet = ParagraphStyle(

        "FormalBullet",

        parent=body,

        leftIndent=7 * mm,

        firstLineIndent=-5 * mm,

        spaceAfter=5,

        alignment=TA_LEFT
    )


    small = ParagraphStyle(

        "FormalSmall",

        parent=body_no_indent,

        fontSize=9,

        leading=14
    )


    question = ParagraphStyle(

        "FormalQuestion",

        parent=body_no_indent,

        fontSize=10.5,

        leading=18,

        spaceAfter=5
    )


    worksheet_question = ParagraphStyle(

        "WorksheetQuestion",

        parent=body_no_indent,

        fontSize=11,

        leading=19,

        spaceAfter=5
    )


    answer = ParagraphStyle(

        "Answer",

        parent=body_no_indent,

        fontSize=10,

        leading=17,

        spaceAfter=4
    )


    return {

        "title": title,

        "subtitle": subtitle,

        "section": section,

        "subsection": subsection,

        "body": body,

        "body_no_indent": body_no_indent,

        "bullet": bullet,

        "small": small,

        "question": question,

        "worksheet_question": worksheet_question,

        "answer": answer,

        "font": font
    }


# =========================================================
# INFO TABLE
# =========================================================

def make_info_table(
    summary,
    styles
):

    font = styles["font"]


    teacher = summary.get(
        "teacher_name",
        ""
    )


    data = [

        [

            paragraph(
                "วิชา\n" + summary.get(
                    "subject",
                    ""
                ),
                styles["small"]
            ),

            paragraph(
                "ระดับชั้น\n" + summary.get(
                    "grade",
                    ""
                ),
                styles["small"]
            )

        ],

        [

            paragraph(
                "เวลา\n" + summary.get(
                    "duration",
                    ""
                ),
                styles["small"]
            ),

            paragraph(
                "ครูผู้สอน\n" + teacher,
                styles["small"]
            )

        ]

    ]


    table = Table(

        data,

        colWidths=[
            82 * mm,
            82 * mm
        ],

        hAlign="CENTER"
    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f7f5ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                colors.HexColor("#d7d2e8")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#ddd8eb")
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
# BUILD PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    styles = create_styles()


    buffer = BytesIO()


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=17 * mm,

        bottomMargin=17 * mm,

        title="เอกสารการเรียนการสอน",

        author=data.get(
            "summary",
            {}
        ).get(
            "teacher_name",
            ""
        ),

        allowSplitting=1
    )


    story = []


    summary = data.get(
        "summary",
        {}
    )


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        lesson = data.get(
            "lesson_plan",
            {}
        )

        content = data.get(
            "teaching_content",
            {}
        )


        story.append(

            Paragraph(
                "แผนการจัดการเรียนรู้",
                styles["title"]
            )
        )


        story.append(

            Paragraph(
                "เรื่อง " +
                esc(
                    summary.get(
                        "topic",
                        ""
                    )
                ),

                styles["subtitle"]
            )
        )


        story.append(

            make_info_table(
                summary,
                styles
            )
        )


        story.append(
            Spacer(
                1,
                5 * mm
            )
        )


        # จุดประสงค์

        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                styles["section"]
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

                    styles["body_no_indent"]

                )
            )


        # เนื้อหา

        story.append(

            Paragraph(
                "2. สาระและเนื้อหาที่ใช้สอน",
                styles["section"]
            )
        )


        intro = content.get(
            "intro",
            ""
        )


        if intro:

            story.append(

                Paragraph(
                    esc(intro),
                    styles["body"]
                )
            )


        story.append(

            Paragraph(
                "สาระสำคัญ",
                styles["subsection"]
            )
        )


        for item in content.get(
            "concepts",
            []
        ):

            story.append(

                Paragraph(
                    "• " + esc(item),
                    styles["bullet"]
                )
            )


        # ตัวอย่าง

        story.append(

            Paragraph(
                "3. ตัวอย่างสำหรับใช้สอน",
                styles["section"]
            )
        )


        for i, example in enumerate(

            content.get(
                "examples",
                []
            ),

            start=1

        ):

            block = [

                Paragraph(

                    f"ตัวอย่างที่ {i} "
                    f"{esc(example.get('title',''))}",

                    styles["subsection"]

                ),

                Paragraph(

                    esc(
                        example.get(
                            "explanation",
                            ""
                        )
                    ),

                    styles["body"]

                )

            ]


            story.append(

                KeepTogether(
                    block
                )
            )


        # คำถามชวนคิด

        story.append(

            Paragraph(
                "4. คำถามชวนคิด",
                styles["section"]
            )
        )


        for i, item in enumerate(

            content.get(
                "thinking_questions",
                []
            ),

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body_no_indent"]

                )
            )


        # ขั้นตอนการสอน

        story.append(

            Paragraph(
                "5. ขั้นตอนการจัดการเรียนรู้",
                styles["section"]
            )
        )


        rows = [

            [

                Paragraph(
                    "เวลา",
                    styles["small"]
                ),

                Paragraph(
                    "กิจกรรม",
                    styles["small"]
                ),

                Paragraph(
                    "รายละเอียด",
                    styles["small"]
                )

            ]

        ]


        for step in lesson.get(
            "steps",
            []
        ):

            rows.append(

                [

                    Paragraph(
                        esc(
                            step.get(
                                "time",
                                ""
                            )
                        ),
                        styles["small"]
                    ),

                    Paragraph(
                        esc(
                            step.get(
                                "title",
                                ""
                            )
                        ),
                        styles["small"]
                    ),

                    Paragraph(
                        esc(
                            step.get(
                                "detail",
                                ""
                            )
                        ),
                        styles["small"]
                    )

                ]

            )


        table = Table(

            rows,

            colWidths=[
                24 * mm,
                42 * mm,
                100 * mm
            ],

            repeatRows=1,

            splitByRow=1
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
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#222222")
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    colors.HexColor("#d5d0df")
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


        story.append(
            table
        )


        # เคล็ดลับ

        story.append(

            Paragraph(
                "6. เคล็ดลับสำหรับครู",
                styles["section"]
            )
        )


        for item in content.get(
            "teacher_tips",
            []
        ):

            story.append(

                Paragraph(
                    "• " + esc(item),
                    styles["bullet"]
                )
            )


        # ประเมิน

        story.append(

            Paragraph(
                "7. การประเมินผล",
                styles["section"]
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

           
