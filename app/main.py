import os
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="AI ครูผู้ช่วย",
    version="1.0.0"
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
# OPENAI MODEL
# =========================================================
# สำคัญ:
# ใช้ gpt-5-mini โดยตรง
# ไม่อ่าน OPENAI_MODEL จาก Render
# เพื่อป้องกันชื่อ gpt-5.6-mini ที่ผิดค้างอยู่
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
จากครู เช่น

"เศษส่วน ป.4 1 ชั่วโมง"

ให้กลายเป็นชุดการสอนพร้อมใช้งาน

ประกอบด้วย:

1. ข้อมูลสรุปบทเรียน
2. จุดประสงค์การเรียนรู้
3. ขั้นตอนการสอน
4. การประเมินผล
5. ใบงาน
6. เฉลยใบงาน
7. แบบทดสอบ
8. เฉลยแบบทดสอบ
9. คำอธิบายคำตอบ


กติกา:

- ใช้ภาษาไทยเป็นหลัก
- วิเคราะห์ระดับชั้นจากคำสั่งของครู
- วิเคราะห์วิชา
- วิเคราะห์หัวข้อ
- วิเคราะห์เวลาเรียน
- เนื้อหาต้องเหมาะกับวัย
- เนื้อหาต้องเหมาะกับระดับชั้น
- ถ้าข้อมูลบางส่วนไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- ห้ามอ้างหลักสูตรเฉพาะที่ไม่แน่ใจ
- ห้ามสร้างข้อมูลมั่ว
- ตรวจสอบความถูกต้องของคำตอบ
- ใบงานต้องสัมพันธ์กับหัวข้อ
- เฉลยใบงานต้องตรงกับโจทย์
- แบบทดสอบต้องตรงกับหัวข้อ
- ปรนัยต้องมี 4 ตัวเลือก
- ปรนัยต้องมีคำตอบถูกเพียงหนึ่งข้อ
- เติมคำต้องมีคำตอบที่ชัดเจน
- คำนวณต้องตรวจตัวเลขและหน่วย
- ประยุกต์ใช้ต้องเป็นสถานการณ์ที่เหมาะกับวัย
- ห้ามสร้างข้อสอบประเภทที่ไม่ได้รับอนุญาต

ตอบเป็น JSON ตาม schema เท่านั้น
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
            detail=(
                "ไม่พบไฟล์ static/index.html "
                "กรุณาตรวจสอบโครงสร้างไฟล์ใน GitHub"
            )
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
        "version": "1.0.0",
        "model": MODEL
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        raise HTTPException(
            status_code=500,
            detail=(
                "ยังไม่ได้ตั้งค่า OPENAI_API_KEY "
                "ใน Render Environment Variables"
            )
        )


    # -----------------------------------------------------
    # QUESTION TYPES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TYPE NAMES
    # -----------------------------------------------------

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


ข้อกำหนด:

- สร้างข้อสอบเฉพาะประเภทที่เลือก
- ถ้าเลือกประเภทเดียว ให้สร้างทั้งหมดเป็นประเภทนั้น
- ถ้าเลือกหลายประเภท ให้กระจายจำนวนข้ออย่างเหมาะสม
- จำนวนข้อสอบต้องเท่ากับที่ผู้ใช้กำหนด
- สร้างใบงานประมาณ 10-20 ข้อ
- ใบงานต้องเหมาะกับระดับชั้น
- เฉลยต้องตรงกับใบงาน
- แบบทดสอบต้องตรงกับหัวข้อ
- ตรวจสอบคำตอบก่อนส่ง
"""


    # -----------------------------------------------------
    # OPENAI CLIENT
    # -----------------------------------------------------

    try:

        client = OpenAI(
            api_key=api_key
        )


        # -------------------------------------------------
        # REQUEST
        # -------------------------------------------------

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


        # -------------------------------------------------
        # JSON
        # -------------------------------------------------

        data = json.loads(
            output_text
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="AI ส่งข้อมูลกลับมาไม่ใช่ JSON ที่ถูกต้อง"
        )


    except Exception as e:

        error_message = str(e)

        # -------------------------------------------------
        # MODEL ERROR
        # -------------------------------------------------

        if "model_not_found" in error_message:

            raise HTTPException(
                status_code=500,
                detail=(
                    "ไม่สามารถใช้โมเดล "
                    f"{MODEL} ได้ "
                    "กรุณาตรวจสอบสิทธิ์ของ OpenAI API Key"
                )
            )


        # -------------------------------------------------
        # GENERAL ERROR
        # -------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=(
                "สร้างชุดการสอนไม่สำเร็จ: "
                + error_message
            )
        )
