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

FONT_FILE = APP_DIR / "THSarabun.ttf"


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

คุณคือผู้ช่วยจัดทำเอกสารการเรียนการสอนสำหรับครูไทย

หน้าที่คือเปลี่ยนคำสั่งสั้น ๆ ของครู
ให้เป็นชุดเอกสารการเรียนการสอนที่พร้อมนำไปใช้จริง

ตัวอย่าง:

"ระบบสุริยะ ป.5 1 ชั่วโมง"

ให้สร้าง:

1. แผนการจัดการเรียนรู้
2. เนื้อหาที่ใช้สอน
3. ตัวอย่างการสอน
4. คำถามชวนคิด
5. ใบงาน
6. แบบทดสอบ
7. เฉลย


กฎสำคัญ:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้นและวัยของนักเรียน
- เนื้อหาต้องเหมาะกับระดับชั้น
- หากข้อมูลไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะหากไม่แน่ใจ
- ห้ามสร้างข้อมูลที่ไม่เกี่ยวข้อง
- ตรวจสอบความถูกต้องของคำถามและคำตอบ
- ตรวจสอบเลข ตัวเลข หน่วย และคำตอบทุกข้อ


ใบงาน:

- สร้างประมาณ 10-20 ข้อ
- คำถามแต่ละข้อควรแตกต่างกัน
- เหมาะสำหรับนักเรียนทำจริง
- มีคำตอบสำหรับใช้ทำเฉลย
- ถ้าเป็นคำถามปลายเปิด ให้เฉลยเป็นแนวคำตอบ


แบบทดสอบ:

สร้างเฉพาะประเภทที่ครูเลือก

ประเภทที่รองรับ:

multiple_choice
fill_blank
calculation
application


multiple_choice:

- มี 4 ตัวเลือก
- ถูกเพียง 1 ตัวเลือก
- ตัวเลือกต้องไม่กำกวม


fill_blank:

- คำตอบต้องชัดเจน


calculation:

- ตรวจสอบตัวเลข
- ตรวจสอบหน่วย
- ตรวจสอบคำตอบ


application:

- เป็นสถานการณ์ที่เหมาะกับวัย
- เชื่อมโยงกับเรื่องที่เรียน


ห้ามสร้างประเภทข้อสอบที่ครูไม่ได้เลือก


การจัดข้อความ:

- อย่าใส่เลขข้อซ้ำซ้อน
- ใช้เลขข้อเป็นตัวเลขอารบิก
- แต่ละข้อเป็นข้อความที่แยกชัดเจน
- ถ้าคำถามยาว ให้แบ่งประโยคอย่างเป็นธรรมชาติ
- ตัวเลือกต้องแยกเป็น ก. ข. ค. ง.


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
            detail="ไม่พบ static/index.html"
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


    invalid = [
        x for x in req.question_types
        if x not in allowed_types
    ]


    if invalid:

        raise HTTPException(
            status_code=400,
            detail="รูปแบบข้อสอบไม่ถูกต้อง: "
                   + ", ".join(invalid)
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


    teacher = req.teacher_name.strip()

    if not teacher:
        teacher = "ไม่ระบุ"


    user_prompt = f"""

ข้อมูลจากครู

ชื่อครู:
{teacher}

คำสั่ง:
{req.prompt}

จำนวนข้อสอบ:
{req.question_count} ข้อ

รูปแบบข้อสอบ:
{selected_types}

ระดับความยาก:
{req.difficulty}


ข้อกำหนด:

1. สร้างชุดเอกสารครบชุด

2. ใบงานประมาณ 10-20 ข้อ

3. แบบทดสอบจำนวน {req.question_count} ข้อ

4. แบบทดสอบต้องใช้เฉพาะประเภทที่เลือก

5. ห้ามสร้างประเภทอื่น

6. ตรวจสอบเลขข้อให้เรียงจาก 1 เป็นต้นไป

7. ตรวจสอบคำตอบทุกข้อ

8. ใช้ภาษาที่เหมาะกับนักเรียน

9. เนื้อหาต้องสัมพันธ์กับหัวข้อ

10. อย่าใส่ Markdown ที่ไม่จำเป็น

11. อย่าใส่เลขข้อซ้ำในข้อความคำถาม
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


        data = json.loads(output_text)


        # บังคับชื่อครูจากข้อมูลของผู้ใช้
        data["summary"]["teacher_name"] = teacher


        return data


    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="AI ส่งข้อมูลกลับมาไม่ใช่ JSON ที่ถูกต้อง"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="สร้างชุดการสอนไม่สำเร็จ: " + str(e)
        )


# =========================================================
# FONT
# =========================================================

def register_thai_font():

    if not FONT_FILE.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_FILE}"
        )


    try:

        pdfmetrics.registerFont(
            TTFont(
                "THSarabun",
                str(FONT_FILE)
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"โหลด THSarabun.ttf ไม่สำเร็จ: {e}"
        )


    return "THSarabun"


# =========================================================
# ESCAPE
# =========================================================

def esc(value):

    if value is None:
        return ""

    return html.escape(
        str(value)
    )


# =========================================================
# TEXT TO PARAGRAPH
# =========================================================

def text_para(
    text,
    style
):

    """
    แปลงข้อความให้เป็น Paragraph
    และรักษาการขึ้นบรรทัดใหม่
    """

    text = str(text or "").strip()

    if not text:
        return Paragraph("", style)


    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if line:
            cleaned.append(
                esc(line)
            )

        else:
            cleaned.append("")


    result = "<br/>".join(
        cleaned
    )

    return Paragraph(
        result,
        style
    )


# =========================================================
# PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    font = register_thai_font()


    buffer = BytesIO()


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        leftMargin=22 * mm,

        rightMargin=22 * mm,

        topMargin=20 * mm,

        bottomMargin=20 * mm,

        title="เอกสารการจัดการเรียนรู้",

        author=data.get(
            "summary",
            {}
        ).get(
            "teacher_name",
            ""
        )
    )


    styles = getSampleStyleSheet()


    # =====================================================
    # STYLES
    # =====================================================

    title_style = ParagraphStyle(

        "DocTitle",

        parent=styles["Title"],

        fontName=font,

        fontSize=22,

        leading=27,

        alignment=TA_CENTER,

        spaceAfter=5 * mm
    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontName=font,

        fontSize=15,

        leading=20,

        alignment=TA_CENTER,

        spaceAfter=8 * mm
    )


    heading_style = ParagraphStyle(

        "Heading",

        parent=styles["Heading1"],

        fontName=font,

        fontSize=17,

        leading=22,

        alignment=TA_LEFT,

        spaceBefore=7 * mm,

        spaceAfter=4 * mm
    )


    subheading_style = ParagraphStyle(

        "SubHeading",

        parent=styles["Heading2"],

        fontName=font,

        fontSize=15,

        leading=20,

        spaceBefore=5 * mm,

        spaceAfter=3 * mm
    )


    body_style = ParagraphStyle(

        "Body",

        parent=styles["BodyText"],

        fontName=font,

        fontSize=14,

        leading=21,

        alignment=TA_LEFT,

        firstLineIndent=8 * mm,

        spaceAfter=4 * mm
    )


    body_no_indent = ParagraphStyle(

        "BodyNoIndent",

        parent=body_style,

        firstLineIndent=0,

        spaceAfter=4 * mm
    )


    question_style = ParagraphStyle(

        "Question",

        parent=body_style,

        fontSize=15,

        leading=22,

        firstLineIndent=0,

        spaceBefore=5 * mm,

        spaceAfter=2 * mm
    )


    answer_style = ParagraphStyle(

        "Answer",

        parent=body_style,

        fontSize=14,

        leading=21,

        firstLineIndent=8 * mm,

        spaceAfter=3 * mm
    )


    option_style = ParagraphStyle(

        "Option",

        parent=body_style,

        fontSize=14,

        leading=21,

        leftIndent=14 * mm,

        firstLineIndent=0,

        spaceAfter=1.5 * mm
    )


    small_style = ParagraphStyle(

        "Small",

        parent=body_style,

        fontSize=12,

        leading=17,

        firstLineIndent=0
    )


    info_style = ParagraphStyle(

        "Info",

        parent=body_style,

        fontSize=14,

        leading=21,

        firstLineIndent=0,

        spaceAfter=2 * mm
    )


    story = []


    summary = data["summary"]


    teacher_name = summary.get(
        "teacher_name",
        "ไม่ระบุ"
    )


    # =====================================================
    # DOCUMENT HEADER
    # =====================================================

    def add_document_header(
        title
    ):

        story.append(
            Paragraph(
                esc(title),
                title_style
            )
        )


        story.append(
            Paragraph(
                f"เรื่อง {esc(summary['topic'])}",
                subtitle_style
            )
        )


        story.append(
            Paragraph(
                f"วิชา {esc(summary['subject'])}",
                info_style
            )
        )


        story.append(
            Paragraph(
                f"ระดับชั้น {esc(summary['grade'])}",
                info_style
            )
        )


        story.append(
            Paragraph(
                f"เวลา {esc(summary['duration'])}",
                info_style
            )
        )


        story.append(
            Paragraph(
                f"ครูผู้สอน {esc(teacher_name)}",
                info_style
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


        add_document_header(
            "แผนการจัดการเรียนรู้"
        )


        story.append(
            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                heading_style
            )
        )


        for item in lesson["objective"]:

            story.append(
                Paragraph(
                    "• " + esc(item),
                    body_no_indent
                )
            )


        story.append(
            Paragraph(
                "2. เนื้อหาที่ใช้สอน",
                heading_style
            )
        )


        story.append(
            text_para(
                content["intro"],
                body_style
            )
        )


        story.append(
            Paragraph(
                "สาระสำคัญ",
                subheading_style
            )
        )


        for item in content["concepts"]:

            story.append(
                text_para(
                    item,
                    body_style
                )
            )


        story.append(
            Paragraph(
                "ตัวอย่างสำหรับใช้สอน",
                subheading_style
            )
        )


        for example in content["examples"]:

            story.append(
                KeepTogether([

                    Paragraph(
                        esc(
                            example["title"]
                        ),
                        subheading_style
                    ),

                    text_para(
                        example["explanation"],
                        body_style
                    )
                ])
            )


        story.append(
            Paragraph(
                "คำถามชวนคิด",
                subheading_style
            )
        )


        for item in content["thinking_questions"]:

            story.append(
                Paragraph(
                    "• " + esc(item),
                    body_no_indent
                )
            )


        story.append(
            Paragraph(
                "3. ขั้นตอนการจัดการเรียนรู้",
                heading_style
            )
        )


        for index, step in enumerate(
            lesson["steps"],
            start=1
        ):

            story.append(
                Paragraph(
                    f"{index}. "
                    f"{esc(step['title'])}",
                    subheading_style
                )
            )


            story.append(
                Paragraph(
                    f"เวลา {esc(step['time'])}",
                    small_style
                )
            )


            story.append(
                text_para(
                    step["detail"],
                    body_style
                )
            )


        story.append(
            Paragraph(
                "4. เคล็ดลับสำหรับครู",
                heading_style
            )
        )


        for item in content["teacher_tips"]:

            story.append(
                Paragraph(
                    "• " + esc(item),
                    body_no_indent
                )
            )


        story.append(
            Paragraph(
                "5. การประเมินผล",
                heading_style
            )
        )


        story.append(
            text_para(
                lesson["assessment"],
                body_style
            )
        )


    # =====================================================
    # WORKSHEET
    # =====================================================

    def add_worksheet():

        story.append(PageBreak())


        add_document_header(
            "ใบงาน"
        )


        story.append(
            Paragraph(
                "ชื่อ-สกุล ................................................................................................",
                body_no_indent
            )
        )


        story.append(
            Paragraph(
                "ชั้น ............................ เลขที่ ............................ วันที่ ............................",
                body_no_indent
            )
        )


        story.append(
            Paragraph(
                "คำชี้แจง",
                heading_style
            )
        )


        story.append(
            Paragraph(
                "ให้นักเรียนอ่านคำถามแต่ละข้อ และเขียนคำตอบลงในพื้นที่ที่กำหนด",
                body_no_indent
            )
        )


        for item in data["worksheet"]:

            no = item["no"]

            question = str(
                item["question"] or ""
            ).strip()


            # ---------------------------------------------
            # ข้อคำถาม
            # ---------------------------------------------

            question_block = Paragraph(
                f"<b>ข้อ {no}</b>&nbsp;&nbsp;{esc(question)}",
                question_style
            )


            # ---------------------------------------------
            # คำตอบ 1 บรรทัด
            # ---------------------------------------------

            answer_block = Paragraph(
                "คำตอบ................................................................................................................",
                answer_style
            )


            story.append(
                KeepTogether([
                    question_block,
                    answer_block
                ])
            )


            story.append(
                Spacer(
                    1,
                    2 * mm
                )
            )


    # =====================================================
    # QUIZ
    # =====================================================

    def add_quiz():

        story.append(PageBreak())


        add_document_header(
            "แบบทดสอบ"
        )


        story.append(
            Paragraph(
                "ชื่อ-สกุล ................................................................................................",
                body_no_indent
            )
        )


        story.append(
            Paragraph(
                "ชั้น ............................ เลขที่ ............................ วันที่ ............................",
                body_no_indent
            )


        )


        story.append(
            Paragraph(
                "คำชี้แจง",
                heading_style
            )
        )


        story.append(
            Paragraph(
                "ให้นักเรียนทำแบบทดสอบทุกข้อ และเลือกหรือเขียนคำตอบให้ถูกต้อง",
                body_no_indent
            )
        )


        for item in data["quiz"]:

            no = item["no"]

            question = item["question"]

            block = []


            block.append(
                Paragraph(
                    f"<b>ข้อ {no}</b>&nbsp;&nbsp;{esc(question)}",
                    question_style
                )
            )


            options = item.get(
                "options",
                []
            )


            if options:

                letters = [
                    "ก.",
                    "ข.",
                    "ค.",
                    "ง."
                ]


                for i, option in enumerate(
                    options
                ):

                    if i < len(letters):

                        letter = letters[i]

                    else:

                        letter = f"{i + 1}."


                    block.append(
                        Paragraph(
                            f"{letter}&nbsp;&nbsp;{esc(option)}",
                            option_style
                        )
                    )


            else:

                block.append(
                    Paragraph(
                        "คำตอบ................................................................................................",
                        answer_style
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


        add_document_header(
            "เฉลย"
        )


        story.append(
            Paragraph(
                "เฉลยใบงาน",
                heading_style
            )
        )


        for item in data["worksheet"]:

            story.append(
                Paragraph(
                    f"<b>ข้อ {item['no']}</b>",
                    question_style
                )
            )


            story.append(
                text_para(
                    item["answer"],
                    body_style
                )
            )


        story.append(
            Paragraph(
                "เฉลยแบบทดสอบ",
                heading_style
            )
        )


        for item in data["quiz"]:

            story.append(
                Paragraph(
                    f"<b>ข้อ {item['no']}</b>&nbsp;&nbsp;"
                    f"{esc(item['answer'])}",
                    question_style
                )
            )


            if item.get("explanation"):

                story.append(
                    text_para(
                        item["explanation"],
                        body_style
                    )
                )


    # =====================================================
    # SELECT SECTION
    # =====================================================

    if section == "lesson":

        add_lesson()


    elif section == "worksheet":

        add_worksheet()


    elif section == "quiz":

        add_quiz()


    elif section == "answers":

        add_answers()


    else:

        add_lesson()
        add_worksheet()
        add_quiz()
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
            11
        )


        canvas.setFillColor(
            colors.HexColor(
                "#555555"
            )
        )


        canvas.drawCentredString(

            A4[0] / 2,

            10 * mm,

            f"หน้า {doc.page}"
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


        topic = (
            data
            .get("summary", {})
            .get("topic", "document")
        )


        safe_topic = "".join(

            c for c in topic

            if c.isalnum()
            or c in " _-"

        ).strip()


        if not safe_topic:

            safe_topic = "teacher-pack"


        filename = (
            f"{safe_topic}-{section}.pdf"
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
