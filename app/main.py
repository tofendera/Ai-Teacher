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


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI ครูผู้ช่วย",
    version="1.0.0"
)


# =========================================================
# STATIC FILES
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

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-mini"
)


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

หน้าที่ของคุณคือเปลี่ยนคำสั่งสั้น ๆ ของครู
เช่น

"เศษส่วน ป.4 1 ชั่วโมง"

ให้กลายเป็นชุดการสอนพร้อมใช้งาน

ต้องทำสิ่งต่อไปนี้:

1. วิเคราะห์ระดับชั้น
2. วิเคราะห์วิชา
3. วิเคราะห์หัวข้อ
4. วิเคราะห์เวลาเรียน
5. สร้างจุดประสงค์การเรียนรู้
6. สร้างขั้นตอนการสอน
7. สร้างใบงาน
8. สร้างเฉลย
9. สร้างแบบทดสอบ

กติกา:

- ใช้ภาษาไทยเป็นหลัก
- ถ้าผู้ใช้ระบุภาษาอังกฤษ ให้สามารถสร้างเนื้อหาภาษาอังกฤษได้
- เนื้อหาต้องเหมาะกับวัยและระดับชั้น
- ถ้าข้อมูลบางส่วนไม่ชัด ให้ใช้บริบทที่สมเหตุสมผล
- อย่าอ้างหลักสูตรเฉพาะที่ไม่แน่ใจ
- อย่าสร้างข้อมูลมั่ว
- ใบงานและเฉลยต้องตรงกัน
- แบบทดสอบต้องตรงกับหัวข้อ
- ปรนัยต้องมี 4 ตัวเลือก
- ปรนัยต้องมีคำตอบถูกเพียงหนึ่งข้อ
- เติมคำต้องมีคำตอบที่ชัดเจน
- โจทย์คำนวณต้องตรวจคำตอบ ตัวเลข และหน่วย
- โจทย์ประยุกต์ต้องเหมาะกับวัย
- ห้ามสร้างประเภทข้อสอบที่ผู้ใช้ไม่ได้เลือก

ตอบตาม JSON Schema ที่กำหนดเท่านั้น
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
# HOME PAGE
# =========================================================

@app.get("/")
def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "ไม่พบไฟล์ static/index.html "
                f"ที่ {index_file}"
            )
        )

    return FileResponse(
        str(index_file),
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
        "version": "1.0.0"
    }


# =========================================================
# GENERATE LESSON
# =========================================================

@app.post("/api/generate")
def generate(req: GenerateRequest):

    # -----------------------------------------------------
    # CHECK API KEY
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
    # CHECK QUESTION TYPES
    # -----------------------------------------------------

    if not req.question_types:

        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ"
        )


    allowed_types = {
        "multiple_choice",
        "fill_blank",
        "calculation",
        "application"
    }


    for question_type in req.question_types:

        if question_type not in allowed_types:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"รูปแบบข้อสอบไม่ถูกต้อง: "
                    f"{question_type}"
                )
            )


    # -----------------------------------------------------
    # QUESTION TYPE NAMES
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


รูปแบบข้อสอบที่อนุญาตเท่านั้น:

{selected_types}


ระดับความยาก:

{req.difficulty}


กติกาเพิ่มเติม:

- ห้ามสร้างประเภทข้อสอบที่ไม่ได้เลือก
- ถ้าเลือกประเภทเดียว ให้สร้างข้อสอบทั้งหมดเป็นประเภทนั้น
- ถ้าเลือกหลายประเภท ให้กระจายข้อสอบอย่างสมดุล
- จำนวนข้อสอบต้องใกล้เคียงกับจำนวนที่กำหนด
- ใบงานควรมีประมาณ 10-20 ข้อ
- ใบงานต้องสอดคล้องกับหัวข้อ
- เฉลยต้องตรวจสอบความถูกต้อง
"""


    # -----------------------------------------------------
    # OPENAI CLIENT
    # -----------------------------------------------------

    try:

        client = OpenAI(
            api_key=api_key
        )


        # -------------------------------------------------
        # OPENAI REQUEST
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
        # GET OUTPUT
        # -------------------------------------------------

        output_text = response.output_text


        if not output_text:

            raise Exception(
                "AI ไม่ส่งข้อมูลกลับมา"
            )


        # -------------------------------------------------
        # PARSE JSON
        # -------------------------------------------------

        data = json.loads(
            output_text
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail=(
                "AI ส่งข้อมูลกลับมาไม่ใช่ JSON "
                "ที่ถูกต้อง"
            )
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "สร้างชุดการสอนไม่สำเร็จ: "
                f"{str(e)}"
            )
        )
