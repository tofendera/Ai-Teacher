import os
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="AI ครูผู้ช่วย", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-mini")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=500)
    question_count: int = Field(default=10, ge=5, le=30)
    question_types: list[str] = Field(default_factory=lambda: ["multiple_choice"])
    difficulty: str = "mixed"


SYSTEM_PROMPT = """
คุณคือ AI ครูผู้ช่วยสำหรับครูไทย
หน้าที่คือเปลี่ยนคำสั่งสั้นๆ ของครู เช่น "เศษส่วน ป.4 1 ชั่วโมง"
ให้เป็นชุดการสอนพร้อมใช้

กติกา:
- ใช้ภาษาไทย
- วิเคราะห์ระดับชั้น วิชา หัวข้อ และเวลาเรียนจากข้อความของครู
- ถ้าข้อมูลบางส่วนไม่ชัด ให้ใช้บริบทที่สมเหตุสมผลและระบุไว้ในผลลัพธ์
- สร้างแผนการสอน ใบงาน เฉลย และแบบทดสอบ
- เนื้อหาต้องเหมาะกับวัยและระดับชั้น
- ใบงานและเฉลยต้องสอดคล้องกันทุกข้อ
- แบบทดสอบต้องสร้างเฉพาะประเภทที่ผู้ใช้เลือก
- สำหรับปรนัย ให้มี 4 ตัวเลือกและมีคำตอบถูกเพียงหนึ่งข้อ
- เติมคำต้องมีคำตอบชัดเจน
- คำนวณต้องตรวจเลขและหน่วย
- ประยุกต์ใช้ต้องเป็นโจทย์สถานการณ์ที่เหมาะกับวัย
- อย่าสร้างข้อมูลมั่วหรืออ้างหลักสูตรเฉพาะที่ไม่แน่ใจ
- ตอบเป็น JSON เท่านั้นตาม schema ที่กำหนด
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grade": {"type": "string"},
                "subject": {"type": "string"},
                "topic": {"type": "string"},
                "duration": {"type": "string"}
            },
            "required": ["grade", "subject", "topic", "duration"]
        },
        "lesson_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objective": {"type": "array", "items": {"type": "string"}},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "time": {"type": "string"},
                            "title": {"type": "string"},
                            "detail": {"type": "string"}
                        },
                        "required": ["time", "title", "detail"]
                    }
                },
                "assessment": {"type": "string"}
            },
            "required": ["objective", "steps", "assessment"]
        },
        "worksheet": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "no": {"type": "integer"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["no", "question", "answer"]
            }
        },
        "quiz": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "no": {"type": "integer"},
                    "type": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["no", "type", "question", "options", "answer", "explanation"]
            }
        }
    },
    "required": ["summary", "lesson_plan", "worksheet", "quiz"]
}


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Environment Variables"
        )

    if not req.question_types:
        raise HTTPException(status_code=400, detail="กรุณาเลือกรูปแบบข้อสอบอย่างน้อย 1 แบบ")

    allowed = {"multiple_choice", "fill_blank", "calculation", "application"}
    if any(t not in allowed for t in req.question_types):
        raise HTTPException(status_code=400, detail="รูปแบบข้อสอบไม่ถูกต้อง")

    client = OpenAI(api_key=api_key)

    type_names = {
        "multiple_choice": "ปรนัย",
        "fill_blank": "เติมคำ",
        "calculation": "คำนวณ",
        "application": "ประยุกต์ใช้"
    }

    selected = ", ".join(type_names[t] for t in req.question_types)

    user_prompt = f"""
คำสั่งจากครู:
{req.prompt}

จำนวนข้อสอบ: {req.question_count} ข้อ
รูปแบบข้อสอบที่อนุญาตเท่านั้น: {selected}
ระดับความยาก: {req.difficulty}

กระจายจำนวนข้อสอบให้เหมาะสมตามประเภทที่เลือก
ห้ามสร้างประเภทที่ไม่ได้เลือก
ถ้าเลือกประเภทเดียว ให้ข้อสอบทั้งหมดเป็นประเภทนั้น
ถ้าเลือกหลายประเภท ให้กระจายอย่างสมดุลโดยประมาณ

สร้างใบงานประมาณ 10-20 ข้อ โดยให้เหมาะกับหัวข้อ
"""

    try:
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
        data = json.loads(response.output_text)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"สร้างชุดการสอนไม่สำเร็จ: {str(e)}")
