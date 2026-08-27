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
from reportlab.pdfgen import canvas


# =========================================================
# PATH
# =========================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# IMPORTANT:
# Font อยู่ข้าง main.py
FONT_FILE = APP_DIR / "NotoSansThai-Regular.ttf"


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
# MODEL
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

คุณคือผู้ช่วยจัดเตรียมเอกสารการเรียนการสอนสำหรับครูไทย

หน้าที่คือเปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็นชุดเอกสารการเรียนการสอนที่นำไปใช้จริงได้

ตัวอย่าง:

"ระบบสุริยะ ป.5 1 ชั่วโมง"

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


หลักการเขียน:

- ใช้ภาษาไทยเป็นหลัก
- เหมาะกับระดับชั้น
- เหมาะกับวัย
- เหมาะกับเวลาเรียน
- ใช้ภาษาที่ครูสามารถนำไปสอนได้จริง
- แบ่งเนื้อหาอย่างเป็นระบบ
- ไม่เขียนทุกอย่างเป็นข้อความยาวก้อนเดียว
- หลีกเลี่ยงข้อมูลที่ไม่แน่ใจ
- ตรวจสอบความถูกต้องก่อนส่ง


เนื้อหาการสอนควรประกอบด้วย:

- บทนำ
- แนวคิดสำคัญ
- คำอธิบาย
- ตัวอย่าง
- วิธีอธิบายให้นักเรียนเข้าใจ
- คำถามชวนคิด
- เคล็ดลับสำหรับครู


ใบงาน:

- เหมาะกับระดับชั้น
- มีคำถามชัดเจน
- มีความหลากหลาย
- มีพื้นที่สำหรับตอบ
- ไม่จำเป็นต้องซ้ำกับแบบทดสอบ


แบบทดสอบรองรับ:

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
- เชื่อมโยงชีวิตประจำวัน
- ต้องสัมพันธ์กับบทเรียน


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
        "font": str(FONT_FILE),
        "font_exists": FONT_FILE.exists()
    }


# =========================================================
# GENERATE TEACHER PACK
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
        x
        for x in req.question_types
        if x not in allowed_types
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

        type_names[x]
        for x in req.question_types
    )


    teacher_name = req.teacher_name.strip()

    if not teacher_name:

        teacher_name = "ไม่ระบุ"


    user_prompt = f"""

คำสั่งจากครู:

{req.prompt}


ชื่อครูผู้สอน:

{teacher_name}


จำนวนข้อสอบ:

{req.question_count} ข้อ


รูปแบบข้อสอบ:

{selected_types}


ระดับความยาก:

{req.difficulty}


ข้อกำหนด:

- สร้างเอกสารการเรียนการสอนครบชุด
- ใช้ชื่อครูผู้สอนเป็น "{teacher_name}"
- ใบงานประมาณ 10-20 ข้อ
- แบบทดสอบจำนวน {req.question_count} ข้อ
- ใช้เฉพาะประเภทข้อสอบที่เลือก
- ห้ามสร้างประเภทอื่น
- ตรวจสอบคำตอบทุกข้อ
- ตรวจสอบตัวเลขและหน่วย
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
# REGISTER THAI FONT
# =========================================================

def register_thai_font():

    if not FONT_FILE.exists():

        raise RuntimeError(
            "ไม่พบไฟล์ Font: "
            f"{FONT_FILE}"
        )


    font_name = "ThaiRegularV15"


    if font_name not in pdfmetrics.getRegisteredFontNames():

        pdfmetrics.registerFont(

            TTFont(
                font_name,
                str(FONT_FILE)
            )
        )


    return font_name


# =========================================================
# ESCAPE HTML
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
# PARAGRAPH
# =========================================================

def P(
    text,
    style
):

    return Paragraph(

        esc(text)
        .replace("\n", "<br/>"),

        style
    )


# =========================================================
# PAGE NUMBER CANVAS
# =========================================================

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

        self.saved_page_states = []


    def showPage(self):

        self.saved_page_states.append(
            dict(self.__dict__)
        )

        self._startPage()


    def save(self):

        page_count = len(
            self.saved_page_states
        )


        for state in self.saved_page_states:

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

        font = register_thai_font()


        self.saveState()


        self.setFont(
            font,
            8
        )


        self.setFillColor(
            colors.HexColor("#777777")
        )


        self.drawCentredString(

            A4[0] / 2,

            8 * mm,

            f"หน้า {self._pageNumber} จาก {page_count}"
        )


        self.restoreState()


# =========================================================
# STYLES
# =========================================================

def get_styles():

    font = register_thai_font()

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "DocTitle",

        parent=styles["Title"],

        fontName=font,

        fontSize=18,

        leading=24,

        alignment=TA_CENTER,

        spaceAfter=4,

        textColor=colors.HexColor("#222222")
    )


    subtitle = ParagraphStyle(

        "DocSubtitle",

        parent=styles["Normal"],

        fontName=font,

        fontSize=11,

        leading=18,

        alignment=TA_CENTER,

        spaceAfter=12
    )


    section = ParagraphStyle(

        "Section",

        parent=styles["Heading1"],

        fontName=font,

        fontSize=13,

        leading=20,

        spaceBefore=11,

        spaceAfter=7,

        keepWithNext=True,

        textColor=colors.HexColor("#222222")
    )


    subsection = ParagraphStyle(

        "Subsection",

        parent=styles["Heading2"],

        fontName=font,

        fontSize=11.5,

        leading=18,

        spaceBefore=7,

        spaceAfter=5,

        keepWithNext=True
    )


    body = ParagraphStyle(

        "Body",

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

        "BodyNoIndent",

        parent=body,

        firstLineIndent=0
    )


    bullet = ParagraphStyle(

        "Bullet",

        parent=body,

        leftIndent=8 * mm,

        firstLineIndent=-5 * mm,

        alignment=TA_LEFT,

        spaceAfter=5
    )


    table = ParagraphStyle(

        "Table",

        parent=body_no_indent,

        fontSize=9,

        leading=14,

        alignment=TA_LEFT
    )


    worksheet = ParagraphStyle(

        "Worksheet",

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

        "font": font,

        "title": title,

        "subtitle": subtitle,

        "section": section,

        "subsection": subsection,

        "body": body,

        "body_no_indent": body_no_indent,

        "bullet": bullet,

        "table": table,

        "worksheet": worksheet,

        "answer": answer
    }


# =========================================================
# INFO TABLE
# =========================================================

def info_table(
    summary,
    styles
):

    data = [

        [

            P(
                "วิชา\n" +
                summary.get(
                    "subject",
                    ""
                ),
                styles["table"]
            ),

            P(
                "ระดับชั้น\n" +
                summary.get(
                    "grade",
                    ""
                ),
                styles["table"]
            )

        ],

        [

            P(
                "เวลา\n" +
                summary.get(
                    "duration",
                    ""
                ),
                styles["table"]
            ),

            P(
                "ครูผู้สอน\n" +
                summary.get(
                    "teacher_name",
                    ""
                ),
                styles["table"]
            )

        ]

    ]


    t = Table(

        data,

        colWidths=[
            82 * mm,
            82 * mm
        ],

        hAlign="CENTER"
    )


    t.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f6f4ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                colors.HexColor("#d6d1e5")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#ded9ea")
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
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    return t


# =========================================================
# STUDENT HEADER
# =========================================================

def student_header(
    summary,
    styles,
    title
):

    story = []


    story.append(

        Paragraph(
            title,
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

        info_table(
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


    return story


# =========================================================
# BUILD PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    styles = get_styles()


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

        allowSplitting=True
    )


    story = []


    summary = data.get(
        "summary",
        {}
    )


    lesson = data.get(
        "lesson_plan",
        {}
    )


    content = data.get(
        "teaching_content",
        {}
    )


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        story.extend(

            student_header(
                summary,
                styles,
                "แผนการจัดการเรียนรู้"
            )
        )


        # 1

        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                styles["section"]
            )
        )


        objectives = lesson.get(
            "objective",
            []
        )


        for i, item in enumerate(
            objectives,
            start=1
        ):

            story.append(

                Paragraph(

                    f"{i}. {esc(item)}",

                    styles["body_no_indent"]
                )
            )


        # 2

        story.append(

            Paragraph(
                "2. สาระและเนื้อหาที่ใช้สอน",
                styles["section"]
            )
        )


        if content.get("intro"):

            story.append(

                Paragraph(
                    esc(
                        content["intro"]
                    ),
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


        # 3

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


        # 4

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


        # 5

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
                    styles["table"]
                ),

                Paragraph(
                    "กิจกรรม",
                    styles["table"]
                ),

                Paragraph(
                    "รายละเอียด",
                    styles["table"]
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
                        styles["table"]
                    ),

                    Paragraph(
                        esc(
                            step.get(
                                "title",
                                ""
                            )
                        ),
                        styles["table"]
                    ),

                    Paragraph(
                        esc(
                            step.get(
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

                repeatRows=1,

                splitByRow=True
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
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#d2ccdf")
                    ),

                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#ddd8e6")
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


        # 6

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


        # 7

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
            )
        )


    # =====================================================
    # WORKSHEET
    # =====================================================

    def add_worksheet():

        story.append(PageBreak())


        story.extend(

            student_header(
                summary,
                styles,
                "ใบงาน"
            )
        )


        story.append(

            Paragraph(
                "ชื่อ-สกุล ................................................................................................",
                styles["body_no_indent"]
            )
        )


        story.append(

            Paragraph(
                "ชั้น ............................... เลขที่ ....................... วันที่ ................................",
                styles["body_no_indent"]
            )
        )


        story.append(
            Spacer(
                1,
                3 * mm
            )
        )


        story.append(

            Paragraph(
                "คำชี้แจง",
                styles["section"]
            )
        )


        story.append(

            Paragraph(

                "ให้นักเรียนอ่านคำถามแต่ละข้อ "
                "และเขียนคำตอบลงในพื้นที่ที่กำหนด",

                styles["body"]
            )
        )


        worksheet_items = data.get(
            "worksheet",
            []
        )


        for item in worksheet_items:

            no = item.get(
                "no",
                ""
            )


            question = item.get(
                "question",
                ""
            )


            block = []


            block.append(

                Paragraph(

                    f"<b>ข้อ {esc(no)}.</b> "
                    f"{esc(question)}",

                    styles["worksheet"]
                )
            )


            # เส้นคำตอบ

            block.append(

                Paragraph(
                    "คำตอบ",
                    styles["answer"]
                )
            )


            # พื้นที่ตอบ 2 บรรทัด

            block.append(

                HRFlowable(

                    width="100%",

                    thickness=0.5,

                    color=colors.HexColor("#bdbdbd"),

                    spaceBefore=4,

                    spaceAfter=12
                )
            )


            block.append(

                HRFlowable(

                    width="100%",

                    thickness=0.5,

                    color=colors.HexColor("#bdbdbd"),

                    spaceBefore=0,

                    spaceAfter=14
                )
            )


            story.append(

                KeepTogether(
                    block
                )
            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(PageBreak())


        story.extend(

            student_header(
                summary,
                styles,
                "แบบทดสอบ"
            )
        )


        story.append(

            Paragraph(
                "ชื่อ-สกุล ................................................................................................",
                styles["body_no_indent"]
            )
        )


        story.append(

            Paragraph(
                "ชั้น ............................... เลขที่ .......................",
                styles["body_no_indent"]
            )
        )


        story.append(

            Paragraph(
                "คำชี้แจง: ให้นักเรียนทำแบบทดสอบทุกข้อ "
                "และเลือกหรือเขียนคำตอบให้ถูกต้อง",
                styles["body"]
            )
        )


        type_names = {

            "multiple_choice": "ปรนัย",

            "fill_blank": "เติมคำ",

            "calculation": "คำนวณ",

            "application": "ประยุกต์ใช้"
        }


        for item in data.get(
            "quiz",
            []
        ):

            block = []


            label = type_names.get(

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

                    f"<b>ข้อ {esc(item.get('no',''))} "
                    f"({esc(label)})</b>",

                    styles["worksheet"]
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

                    styles["body_no_indent"]
                )
            )


            options = item.get(
                "options",
                []
            )


            if options:

                for index, option in enumerate(
                    options
                ):

                    letter = chr(
                        65 + index
                    )


                    block.append(

                        Paragraph(

                            f"{letter}. "
                            f"{esc(option)}",

                            styles["body_no_indent"]
                        )
                    )


            else:

                block.append(

                    Spacer(
                        1,
                        5 * mm
                    )
                )


                block.append(

                    HRFlowable(

                        width="100%",

                        thickness=0.5,

                        color=colors.HexColor("#999999"),

                        spaceBefore=3,

                        spaceAfter=12
                    )
                )


            story.append(

                KeepTogether(
                    block
                )
            )


            story.append(
                Spacer(
                    1,
                    3 * mm
                )
            )


    # =====================================================
    # ANSWERS
    # =====================================================

    def add_answers():

        story.append(PageBreak())


        story.extend(

            student_header(
                summary,
                styles,
                "เฉลย"
            )
        )


        story.append(

            Paragraph(
                "เฉลยใบงาน",
                styles["section"]
            )
        )


        for item in data.get(
            "worksheet",
            []
        ):

            story.append(

                Paragraph(

                    f"<b>ข้อ {esc(item.get('no',''))}.</b> "
                    f"{esc(item.get('answer',''))}",

                    styles["answer"]
                )
            )


        story.append(

            Paragraph(
                "เฉลยแบบทดสอบ",
                styles["section"]
            )
        )


        for item in data.get(
            "quiz",
            []
        ):

            block = [

                Paragraph(

                    f"<b>ข้อ {esc(item.get('no',''))}.</b> "
                    f"{esc(item.get('answer',''))}",

                    styles["answer"]
                ),

                Paragraph(

                    esc(
                        item.get(
                            "explanation",
                            ""
                        )
                    ),

                    styles["small"]
                    if "small" in styles
                    else styles["answer"]
                )

            ]


            story.append(

                KeepTogether(
                    block
                )
            )


    # =====================================================
    # SELECT
    # =====================================================

    if section == "all":

        add_lesson()
        add_worksheet()
        add_quiz()
        add_answers()

    elif section == "lesson":

        add_lesson()

    elif section == "worksheet":

        add_worksheet()

    elif section == "quiz":

        add_quiz()

    elif section == "answers":

        add_answers()


    # =====================================================
    # BUILD
    # =====================================================

    doc.build(

        story,

        canvasmaker=NumberedCanvas
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


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "สร้าง PDF ไม่สำเร็จ: "
                + str(e)
            )
        )
