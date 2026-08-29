import os
import json
import html
import re
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
FONT_BOLD_FILE = APP_DIR / "THSarabunBold.ttf"

VISIT_FILE = APP_DIR / "visit_count.txt"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Teacher Pack",
    version="1.6.2"
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


MODEL = "gpt-5-mini"


# =========================================================
# REQUEST
# =========================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        min_length=2,
        max_length=500
    )

    subject: str = Field(
        default="",
        max_length=100,
        description="ชื่อวิชา เช่น ภาษาอังกฤษ, คณิตศาสตร์, วิทยาศาสตร์"
    )

    teacher_name: str = Field(
        default="",
        max_length=100
    )

    # คำศัพท์ / มโนทัศน์ / นิยามสำคัญ
    # ใช้ได้ทุกวิชา ไม่ใช่แค่วิชาภาษา
    key_terms: list[str] = Field(
        default_factory=list,
        description="คำศัพท์ / มโนทัศน์ / นิยามสำคัญที่ต้องใช้ในเนื้อหา"
    )

    # โครงสร้างประโยค / สูตร / ขั้นตอน / กฎสำคัญ
    key_patterns: list[str] = Field(
        default_factory=list,
        description="โครงสร้างประโยค / สูตร / ขั้นตอน / กฎสำคัญที่ต้องใช้"
    )

    learning_focus: str = Field(
        default="",
        max_length=300,
        description="จุดเน้นเพิ่มเติม เช่น เน้นฟัง-พูด, เน้นคำนวณ, เน้นทดลอง"
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
ใช้ได้กับทุกวิชา ไม่จำกัดเฉพาะวิชาใดวิชาหนึ่ง

สร้างเอกสารดังนี้:

1. แผนการจัดการเรียนรู้
2. เนื้อหาที่ใช้สอน
3. ตัวอย่างสำหรับใช้สอน
4. คำถามชวนคิด
5. ใบงาน
6. แบบทดสอบ
7. เฉลย

ข้อกำหนด:

- ใช้ภาษาไทยเป็นหลัก
- เหมาะสมกับระดับชั้น
- เนื้อหาถูกต้อง
- ตรวจสอบตัวเลขและคำตอบ
- ห้ามใส่ Emoji ในข้อมูลที่จะนำไปสร้าง PDF
- ห้ามใช้สัญลักษณ์ตกแต่งที่ไม่จำเป็น
- ใช้ตัวเลขอารบิก
- เลขข้อเรียงจาก 1 เป็นต้นไป

ตัวเลือกข้อสอบปรนัย (options):

- ห้ามใส่ตัวอักษร ก. ข. ค. ง. หรือหมายเลขนำหน้าข้อความ options เอง
  ให้ส่งเฉพาะเนื้อหาของตัวเลือกล้วนๆ เท่านั้น
  เพราะระบบ (PDF และหน้าเว็บ) จะใส่ ก. ข. ค. ง. ให้เองอัตโนมัติ
  ถ้าใส่ซ้ำจะทำให้ตัวอักษรซ้อนกันสองชั้น

คำอธิบายเฉลย (explanation) ในแบบทดสอบ (สำคัญมาก):

- ต้องเขียนเป็นภาษาไทยเสมอ ไม่ว่าวิชาหรือภาษาของโจทย์จะเป็นภาษาใด
  เช่น ข้อสอบวิชาภาษาอังกฤษ ตัวโจทย์และตัวเลือกเป็นภาษาอังกฤษได้
  แต่คำอธิบายเฉลยต้องเขียนอธิบายเป็นภาษาไทยทั้งหมด
  จะแทรกคำศัพท์หรือตัวอย่างประโยคภาษาอังกฤษสั้นๆ ประกอบได้
  แต่ตัวคำอธิบายหลักต้องเป็นภาษาไทย
- ห้ามคัดลอกคำสั่งของโจทย์มาใส่ในคำอธิบาย เช่น ห้ามขึ้นต้นด้วย
  "Choose the correct form", "Select the answer" หรือคำสั่งทำนอง
  เดียวกัน ให้อธิบายเหตุผลตรงๆ เป็นภาษาไทยแทน
- ต้องอธิบายเหตุผลว่าทำไมคำตอบนั้นถูกต้อง เช่น หลักไวยากรณ์
  สูตรที่ใช้ แนวคิดหรือหลักการที่เกี่ยวข้อง ไม่ใช่แค่บอกคำตอบซ้ำ
  ตัวอย่างที่ถูกต้อง: "ใช้ can เพราะประโยคนี้ต้องการสื่อถึง
  ความสามารถ (ability) ของประธาน"
  ตัวอย่างที่ห้ามทำ: "Choose the correct form: (ability)"

กติกาเรื่องคำศัพท์/สูตร/โครงสร้างสำคัญ:

- หากผู้ใช้ระบุ "เนื้อหา/คำศัพท์/มโนทัศน์สำคัญ" หรือ
  "โครงสร้าง/สูตร/รูปแบบสำคัญ" มาให้ ต้องใช้สิ่งเหล่านั้น
  เป็นแกนหลักของเนื้อหาทั้งหมด ห้ามแต่งเนื้อหาอื่นที่ไม่เกี่ยวข้อง
- หากไม่ได้ระบุมา ให้พิจารณาความเหมาะสมกับวิชาและระดับชั้นเอง
- ปรับรูปแบบการอธิบายให้เข้ากับธรรมชาติของแต่ละวิชา เช่น
  วิชาภาษาเน้นคำศัพท์และประโยค
  วิชาคณิตศาสตร์เน้นสูตรและตัวอย่างการคำนวณ
  วิชาวิทยาศาสตร์เน้นนิยามและขั้นตอนการทดลอง
  วิชาสังคม/ประวัติศาสตร์เน้นเหตุการณ์และลำดับเวลา

ความละเอียดของเนื้อหา (สำคัญมาก):

เนื้อหาที่สร้างต้องนำไปใช้สอนได้จริงทันที ไม่ใช่แค่หัวข้อ
หรือแนวทางกว้างๆ แบบไกด์ไลน์ ให้ยึดตามนี้:

- intro: คำนำเข้าสู่เนื้อหา เขียนอย่างน้อย 4-6 ประโยค
  อธิบายว่าเรื่องนี้คืออะไร เกี่ยวข้องกับชีวิตนักเรียนอย่างไร
- concepts (สาระสำคัญ): อย่างน้อย 4 ข้อ แต่ละข้อเขียนอธิบาย
  อย่างน้อย 3-5 ประโยค ไม่ใช่แค่หัวข้อคำเดียวหรือประโยคเดียว
- examples (ตัวอย่างสำหรับใช้สอน): อย่างน้อย 2 ตัวอย่าง
  แต่ละตัวอย่างอธิบายเป็นขั้นตอนที่ครูอ่านแล้วสอนตามได้ทันที
  ความยาวอย่างน้อย 4-6 ประโยค รวมบทสนทนาหรือคำถาม-คำตอบ
  ตัวอย่างที่ครูใช้พูดในห้องเรียนจริง
- thinking_questions: อย่างน้อย 3 ข้อ
- teacher_tips: อย่างน้อย 3 ข้อ เป็นเคล็ดลับที่นำไปใช้ได้จริง
- steps (ขั้นตอนการจัดการเรียนรู้): แต่ละขั้นตอนอธิบายกิจกรรม
  ที่ครูทำและนักเรียนทำอย่างละเอียด อย่างน้อย 3-4 ประโยค
  ระบุคำถามหรือประโยคตัวอย่างที่ใช้จริงในชั้นเรียนถ้าเกี่ยวข้อง
- assessment: อธิบายวิธีประเมินผลอย่างละเอียด อย่างน้อย 3 ประโยค
  ถ้ามีหลายข้อ ให้ขึ้นต้นแต่ละข้อด้วย 1) 2) 3) ตามลำดับ

ใบงาน:

ข้อ 1  คำถาม

คำตอบ

ข้อ 2  คำถาม

คำตอบ

แต่ให้ส่งข้อมูลแต่ละส่วนแยกกันตาม JSON Schema

ห้ามใส่เลขข้อซ้ำในข้อความ question

แบบทดสอบ:

ถ้าเป็นปรนัยให้มี 4 ตัวเลือก
ถูกเพียง 1 ตัวเลือก

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
        str(INDEX_FILE)
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "app": "Teacher Pack",

        "version": "1.6.2",

        "font": FONT_FILE.name,

        "font_exists": FONT_FILE.exists(),

        "font_bold": FONT_BOLD_FILE.name,

        "font_bold_exists": FONT_BOLD_FILE.exists()

    }


# =========================================================
# VISITS
# นับจำนวนครั้งที่มีคนเข้าเว็บ (เก็บลงไฟล์ ไม่ต้องใช้ DB)
# =========================================================

@app.get("/api/visits")
def visits():

    count = 0


    if VISIT_FILE.exists():

        try:

            count = int(
                VISIT_FILE
                .read_text()
                .strip()
                or "0"
            )

        except ValueError:

            count = 0


    count += 1


    try:

        VISIT_FILE.write_text(
            str(count)
        )

    except Exception:

        # ถ้าเขียนไฟล์ไม่ได้ (เช่น เขียนพร้อมกันหลาย request)
        # ให้ยังคืนเลขล่าสุดที่คำนวณได้ ไม่ทำให้ endpoint พัง

        pass


    return {
        "count": count
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
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY"
        )


    allowed_types = {

        "multiple_choice",

        "fill_blank",

        "calculation",

        "application"

    }


    invalid = [

        x for x in req.question_types

        if x not in allowed_types

    ]


    if invalid:

        raise HTTPException(

            status_code=400,

            detail=(
                "รูปแบบข้อสอบไม่ถูกต้อง: "
                + ", ".join(invalid)
            )

        )


    teacher = (
        req.teacher_name.strip()
        or "ไม่ระบุ"
    )


    subject = (
        req.subject.strip()
        or "ไม่ระบุ (ให้พิจารณาจากคำสั่ง)"
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


    # ---------------------------------------------------
    # แปลง key_terms / key_patterns (list) เป็นข้อความ bullet
    # ---------------------------------------------------

    key_terms_text = (

        "\n".join(

            f"- {t.strip()}"

            for t in req.key_terms

            if t.strip()

        )

        or "ไม่ได้ระบุ (ให้ AI พิจารณาความเหมาะสมเอง)"

    )


    key_patterns_text = (

        "\n".join(

            f"- {p.strip()}"

            for p in req.key_patterns

            if p.strip()

        )

        or "ไม่ได้ระบุ"

    )


    learning_focus_text = (
        req.learning_focus.strip()
        or "ไม่ระบุ"
    )


    user_prompt = f"""

วิชา:
{subject}

ชื่อครู:
{teacher}

คำสั่ง:
{req.prompt}

เนื้อหา/คำศัพท์/มโนทัศน์สำคัญที่ต้องใช้เป็นแกนหลัก:
{key_terms_text}

โครงสร้าง/สูตร/รูปแบบสำคัญที่ต้องใช้:
{key_patterns_text}

จุดเน้นการเรียนรู้เพิ่มเติม:
{learning_focus_text}

จำนวนข้อสอบ:
{req.question_count}

ประเภท:
{selected_types}

ระดับความยาก:
{req.difficulty}

สร้างเอกสารให้พร้อมใช้จริง เนื้อหาต้องละเอียดครบถ้วน
ไม่ใช่แค่หัวข้อหรือแนวทางกว้างๆ

ตรวจสอบเลขข้อ

ตรวจสอบตัวเลข

ตรวจสอบคำตอบ

ห้ามใส่ Emoji

ห้ามใส่ ก. ข. ค. ง. นำหน้า options เอง

คำอธิบายเฉลย (explanation) ทุกข้อต้องเป็นภาษาไทยเสมอ
ห้ามคัดลอกคำสั่งของโจทย์ภาษาอังกฤษมาใส่ในคำอธิบาย

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


        if not response.output_text:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        data = json.loads(
            response.output_text
        )


        data[
            "summary"
        ][
            "teacher_name"
        ] = teacher


        # หากผู้ใช้ระบุวิชามาชัดเจน
        # ให้ยึดตามที่ผู้ใช้กรอกเสมอ
        # แทนที่ AI จะเดาเอง

        if req.subject.strip():

            data[
                "summary"
            ][
                "subject"
            ] = req.subject.strip()


        # ---------------------------------------------------
        # กันเหนียว: ถ้า AI ยังใส่ ก./ข./ค./ง. หรือหมายเลข
        # นำหน้า option มาเอง ให้ตัดออกก่อนส่งกลับ
        # ป้องกันตัวอักษรซ้อนกันตอนแสดงผล
        # ---------------------------------------------------

        prefix_pattern = re.compile(

            r"^\s*"
            r"(?:[ก-ฮ]\.|[A-Da-d]\.|\d+[\.\)])"
            r"\s*"

        )


        for item in data.get(
            "quiz",
            []
        ):

            options = item.get(
                "options",
                []
            )

            item["options"] = [

                prefix_pattern.sub(
                    "",
                    opt
                ).strip()

                for opt in options

            ]


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail="AI ส่ง JSON ไม่ถูกต้อง"

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
# REGISTER FONT
# =========================================================

def register_thai_font():

    if not FONT_FILE.exists():

        raise FileNotFoundError(

            f"ไม่พบไฟล์ Font: {FONT_FILE}"

        )


    if not FONT_BOLD_FILE.exists():

        raise FileNotFoundError(

            f"ไม่พบไฟล์ Font ตัวหนา: {FONT_BOLD_FILE}"

        )


    pdfmetrics.registerFont(

        TTFont(
            "THSarabun",
            str(FONT_FILE)
        )

    )


    pdfmetrics.registerFont(

        TTFont(
            "THSarabun-Bold",
            str(FONT_BOLD_FILE)
        )

    )


    return "THSarabun", "THSarabun-Bold"


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_pdf_text(value):

    if value is None:

        return ""


    text = str(value)


    # Emoji / pictographic characters
    text = re.sub(

        r"[\U0001F000-\U0001FAFF]",

        "",

        text

    )


    # สัญลักษณ์ตกแต่งที่อาจทำให้
    # ฟอนต์เก่ามีปัญหา

    replacements = {

        "•": "-",

        "●": "-",

        "▪": "-",

        "▫": "-",

        "◦": "-",

        "—": "-",

        "–": "-",

        "…": "...",

        "✓": "ถูก",

        "✔": "ถูก",

        "✕": "ผิด",

        "×": "x",

        "\u00a0": " "

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    return text


# =========================================================
# FORMAT NUMBERED LIST
# แยกบรรทัดให้ 1) 2) 3) ... ขึ้นบรรทัดใหม่เสมอ
# ใช้กับข้อความยาวๆ ที่ AI อาจส่งมาเป็นพืดเดียว
# =========================================================

def format_numbered_list(text):

    if not text:

        return text


    segments = re.split(

        r"(?=\d+\))",

        str(text)

    )


    segments = [

        s.strip()

        for s in segments

        if s and s.strip()

    ]


    return "\n".join(segments)


# =========================================================
# ESCAPE
# =========================================================

def esc(value):

    return html.escape(

        clean_pdf_text(value)

    )


# =========================================================
# TEXT PARAGRAPH
# =========================================================

def text_para(
    text,
    style
):

    text = clean_pdf_text(
        text
    ).strip()


    if not text:

        return Paragraph(
            "",
            style
        )


    lines = text.splitlines()


    result = []


    for line in lines:

        line = line.strip()


        if line:

            result.append(
                esc(line)
            )

        else:

            result.append(
                "<br/>"
            )


    return Paragraph(

        "<br/>".join(result),

        style

    )


# =========================================================
# PDF
# =========================================================

def build_pdf(
    data,
    section="all"
):

    font, font_bold = register_thai_font()


    buffer = BytesIO()


    summary = data.get(
        "summary",
        {}
    )


    teacher_name = summary.get(

        "teacher_name",

        "ไม่ระบุ"

    )


    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        leftMargin=22 * mm,

        rightMargin=22 * mm,

        topMargin=20 * mm,

        bottomMargin=20 * mm,

        title="เอกสารการจัดการเรียนรู้",

        author=teacher_name

    )


    styles = getSampleStyleSheet()


    # =====================================================
    # STYLES
    # หัวข้อ (title / heading / subheading) ใช้ฟอนต์ตัวหนา
    # เนื้อหาที่เหลือใช้ฟอนต์ปกติ
    # =====================================================

    title_style = ParagraphStyle(

        "DocTitle",

        parent=styles["Title"],

        fontName=font_bold,

        fontSize=22,

        leading=27,

        alignment=TA_CENTER,

        spaceAfter=4 * mm

    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontName=font,

        fontSize=16,

        leading=20,

        alignment=TA_CENTER,

        spaceAfter=5 * mm

    )


    info_line_style = ParagraphStyle(

        "InfoLine",

        parent=styles["Normal"],

        fontName=font,

        fontSize=13,

        leading=18,

        alignment=TA_CENTER,

        textColor=colors.HexColor(
            "#555555"
        ),

        spaceAfter=6 * mm

    )


    heading_style = ParagraphStyle(

        "Heading",

        parent=styles["Heading1"],

        fontName=font_bold,

        fontSize=17,

        leading=22,

        alignment=TA_LEFT,

        spaceBefore=7 * mm,

        spaceAfter=4 * mm

    )


    subheading_style = ParagraphStyle(

        "SubHeading",

        parent=styles["Heading2"],

        fontName=font_bold,

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

        firstLineIndent=0

    )


    # -----------------------------------------------------
    # BULLET STYLE
    # ใช้กับรายการที่ขึ้นต้นด้วย "- " เช่น จุดประสงค์การเรียนรู้
    # และเคล็ดลับสำหรับครู ให้ทุกบรรทัดมีย่อหน้า/การเยื้อง
    # ในสัดส่วนเท่ากันทุกข้อ (hanging indent)
    # -----------------------------------------------------

    bullet_style = ParagraphStyle(

        "Bullet",

        parent=body_style,

        fontName=font,

        fontSize=14,

        leading=21,

        leftIndent=8 * mm,

        firstLineIndent=-8 * mm,

        spaceAfter=3 * mm

    )


    question_style = ParagraphStyle(

        "Question",

        parent=body_style,

        fontName=font,

        fontSize=15,

        leading=22,

        firstLineIndent=0,

        spaceBefore=5 * mm,

        spaceAfter=3 * mm

    )


    answer_style = ParagraphStyle(

        "Answer",

        parent=body_style,

        fontName=font,

        fontSize=14,

        leading=21,

        firstLineIndent=8 * mm,

        spaceAfter=3 * mm

    )


    option_style = ParagraphStyle(

        "Option",

        parent=body_style,

        fontName=font,

        fontSize=14,

        leading=21,

        leftIndent=14 * mm,

        firstLineIndent=0,

        spaceAfter=1.5 * mm

    )


    story = []


    # =====================================================
    # HEADER
    # วิชา / ระดับชั้น / เวลา / ครูผู้สอน อยู่บรรทัดเดียวกัน
    # =====================================================

    def add_header(title):

        story.append(

            Paragraph(
                esc(title),
                title_style
            )

        )


        story.append(

            Paragraph(

                "เรื่อง " +
                esc(summary.get(
                    "topic",
                    ""
                )),

                subtitle_style

            )

        )


        info_line = (

            "วิชา " +
            esc(summary.get(
                "subject",
                ""
            )) +

            "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;" +

            "ระดับชั้น " +
            esc(summary.get(
                "grade",
                ""
            )) +

            "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;" +

            "เวลา " +
            esc(summary.get(
                "duration",
                ""
            )) +

            "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;" +

            "ครูผู้สอน " +
            esc(teacher_name)

        )


        story.append(

            Paragraph(

                info_line,

                info_line_style

            )

        )


        story.append(
            Spacer(1, 4 * mm)
        )


    # =====================================================
    # LESSON
    # =====================================================

    def add_lesson():

        lesson = data[
            "lesson_plan"
        ]

        content = data[
            "teaching_content"
        ]


        add_header(
            "แผนการจัดการเรียนรู้"
        )


        story.append(

            Paragraph(
                "1. จุดประสงค์การเรียนรู้",
                heading_style
            )

        )


        for item in lesson[
            "objective"
        ]:

            story.append(

                Paragraph(

                    "- " + esc(item),

                    bullet_style

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


        for item in content[
            "concepts"
        ]:

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


        for item in content[
            "examples"
        ]:

            story.append(

                Paragraph(

                    esc(item["title"]),

                    subheading_style

                )

            )


            story.append(

                text_para(

                    item["explanation"],

                    body_style

                )

            )


        story.append(

            Paragraph(

                "คำถามชวนคิด",

                subheading_style

            )

        )


        for item in content[
            "thinking_questions"
        ]:

            story.append(

                Paragraph(

                    "- " + esc(item),

                    body_no_indent

                )

            )


        story.append(

            Paragraph(

                "3. ขั้นตอนการจัดการเรียนรู้",

                heading_style

            )

        )


        for i, step in enumerate(

            lesson["steps"],

            start=1

        ):

            story.append(

                Paragraph(

                    f"{i}. " +
                    esc(step["title"]),

                    subheading_style

                )

            )


            story.append(

                Paragraph(

                    "เวลา " +
                    esc(step["time"]),

                    body_no_indent

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


        for item in content[
            "teacher_tips"
        ]:

            story.append(

                Paragraph(

                    "- " + esc(item),

                    bullet_style

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

                format_numbered_list(
                    lesson["assessment"]
                ),

                body_style

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


        for item in data[
            "worksheet"
        ]:

            no = item["no"]


            question = clean_pdf_text(

                item["question"]

            ).strip()


            # =============================================
            # รูปแบบที่ต้องการ
            # =============================================

            question_block = Paragraph(

                f"<font name='{font_bold}'>"
                f"ข้อ {no}"
                f"</font>"
                f"&nbsp;&nbsp;"
                f"{esc(question)}",

                question_style

            )


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

        story.append(
            PageBreak()
        )


        add_header(
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


        for item in data[
            "quiz"
        ]:

            no = item["no"]

            question = item[
                "question"
            ]


            block = []


            block.append(

                Paragraph(

                    f"<font name='{font_bold}'>"
                    f"ข้อ {no}"
                    f"</font>"
                    f"&nbsp;&nbsp;"
                    f"{esc(question)}",

                    question_style

                )

            )


            options = item.get(
                "options",
                []
            )


            letters = [
                "ก.",
                "ข.",
                "ค.",
                "ง."
            ]


            for i, option in enumerate(
                options
            ):

                letter = (

                    letters[i]

                    if i < len(letters)

                    else f"{i + 1}."

                )


                block.append(

                    Paragraph(

                        f"{letter}"
                        f"&nbsp;&nbsp;"
                        f"{esc(option)}",

                        option_style

                    )

                )


            if not options:

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

        story.append(
            PageBreak()
        )


        add_header(
            "เฉลย"
        )


        story.append(

            Paragraph(

                "เฉลยใบงาน",

                heading_style

            )

        )


        for item in data[
            "worksheet"
        ]:

            story.append(

                Paragraph(

                    f"<font name='{font_bold}'>"
                    f"ข้อ {item['no']}"
                    f"</font>",

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


        for item in data[
            "quiz"
        ]:

            story.append(

                Paragraph(

                    f"<font name='{font_bold}'>"
                    f"ข้อ {item['no']}"
                    f"</font>"
                    f"&nbsp;&nbsp;"
                    f"{esc(item['answer'])}",

                    question_style

                )

            )


            if item.get(
                "explanation"
            ):

                story.append(

                    text_para(

                        item[
                            "explanation"
                        ],

                        body_style

                    )

                )


    # =====================================================
    # SECTION
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
# PDF API
# =========================================================

@app.post("/api/pdf")
def create_pdf(
    data: dict,
    section: str = "all"
):

    allowed = {

        "all",
        "lesson",
        "worksheet",
        "quiz",
        "answers"

    }


    if section not in allowed:

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
            .get("topic", "teacher-pack")

        )


        # ป้องกันชื่อไฟล์มีอักขระแปลก

        filename = "teacher-pack.pdf"


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
