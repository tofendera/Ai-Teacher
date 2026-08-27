import os
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib import colors


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
PDF_DIR = BASE_DIR / "generated"

PDF_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FONT
# =========================================================

FONT_REGULAR = BASE_DIR / "THSarabun.ttf"
FONT_BOLD = BASE_DIR / "THSarabunBold.ttf"


app = FastAPI(
    title="Ai-Teacher",
    version="1.5"
)


def register_fonts():

    if not FONT_REGULAR.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_REGULAR}"
        )

    if not FONT_BOLD.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Font: {FONT_BOLD}"
        )

    pdfmetrics.registerFont(
        TTFont(
            "THSarabun",
            str(FONT_REGULAR)
        )
    )

    pdfmetrics.registerFont(
        TTFont(
            "THSarabunBold",
            str(FONT_BOLD)
        )
    )


try:

    register_fonts()

    FONT_ERROR = None

except Exception as e:

    FONT_ERROR = str(e)


# =========================================================
# REQUEST
# =========================================================

class GenerateRequest(BaseModel):

    prompt: str

    teacher_name: str = ""

    question_count: int = 10

    question_types: list[str] = [
        "multiple_choice"
    ]

    difficulty: str = "mixed"


# =========================================================
# HELPERS
# =========================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def get_openai():

    key = os.getenv("OPENAI_API_KEY")

    if not key:

        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY"
        )

    return OpenAI(api_key=key)


def normalize_data(
    data: dict,
    req: GenerateRequest
):

    summary = data.get("summary") or {}

    summary["grade"] = (
        clean_text(summary.get("grade"))
        or "ไม่ระบุ"
    )

    summary["subject"] = (
        clean_text(summary.get("subject"))
        or "ไม่ระบุ"
    )

    summary["topic"] = (
        clean_text(summary.get("topic"))
        or req.prompt
    )

    summary["duration"] = (
        clean_text(summary.get("duration"))
        or "1 ชั่วโมง"
    )

    summary["teacher"] = (
        clean_text(req.teacher_name)
        or "ไม่ระบุ"
    )


    lesson = data.get("lesson_plan") or {}

    lesson["objective"] = (
        lesson.get("objective")
        or []
    )

    lesson["steps"] = (
        lesson.get("steps")
        or []
    )

    lesson["assessment"] = clean_text(
        lesson.get("assessment")
    )


    worksheet = data.get("worksheet") or []

    quiz = data.get("quiz") or []


    # -----------------------------------------------------
    # WORKSHEET NUMBER
    # -----------------------------------------------------

    for i, q in enumerate(
        worksheet,
        1
    ):

        q["no"] = i

        q["question"] = clean_text(
            q.get("question")
        )

        q["answer"] = clean_text(
            q.get("answer")
        )

        q["options"] = (
            q.get("options")
            or []
        )


    # -----------------------------------------------------
    # QUIZ NUMBER
    # -----------------------------------------------------

    for i, q in enumerate(
        quiz,
        1
    ):

        q["no"] = i

        q["question"] = clean_text(
            q.get("question")
        )

        q["answer"] = clean_text(
            q.get("answer")
        )

        q["explanation"] = clean_text(
            q.get("explanation")
        )

        q["options"] = (
            q.get("options")
            or []
        )


    return {

        "summary": summary,

        "lesson_plan": lesson,

        "worksheet": worksheet,

        "quiz": quiz,

    }


# =========================================================
# PDF STYLE
# =========================================================

def pstyle(
    name,
    size=16,
    leading=24,
    bold=False,
    align=TA_LEFT,
    first=0,
    space_after=6,
    space_before=0
):

    return ParagraphStyle(

        name=name,

        fontName=(
            "THSarabunBold"
            if bold
            else "THSarabun"
        ),

        fontSize=size,

        leading=leading,

        alignment=align,

        firstLineIndent=first,

        spaceAfter=space_after,

        spaceBefore=space_before,

        textColor=colors.black,
    )


def safe_para(
    text: str,
    style
):

    text = clean_text(text)

    text = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    text = text.replace(
        "\n",
        "<br/>"
    )

    return Paragraph(
        text,
        style
    )


# =========================================================
# LESSON PDF
# =========================================================

def build_lesson_pdf(
    data: dict,
    path: Path
):

    if FONT_ERROR:

        raise RuntimeError(
            FONT_ERROR
        )


    summary = data["summary"]

    lesson = data["lesson_plan"]

    teacher = (
        summary.get("teacher")
        or "ไม่ระบุ"
    )


    doc = SimpleDocTemplate(

        str(path),

        pagesize=A4,

        rightMargin=22 * mm,

        leftMargin=22 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"แผนการจัดการเรียนรู้ - "
            f"{summary['topic']}"
        ),

        author=teacher,
    )


    title = pstyle(
        "T",
        25,
        30,
        True,
        TA_CENTER,
        space_after=2
    )


    subtitle = pstyle(
        "ST",
        18,
        23,
        False,
        TA_CENTER,
        space_after=13
    )


    meta = pstyle(
        "M",
        16,
        21,
        False,
        TA_CENTER,
        space_after=15
    )


    h1 = pstyle(
        "H1",
        19,
        24,
        True,
        TA_LEFT,
        space_before=7,
        space_after=8
    )


    h2 = pstyle(
        "H2",
        17,
        22,
        True,
        TA_LEFT,
        space_before=6,
        space_after=7
    )


    body = pstyle(
        "B",
        16,
        23,
        False,
        TA_LEFT,
        first=9,
        space_after=7
    )


    bullet = pstyle(
        "BL",
        16,
        23,
        False,
        TA_LEFT,
        first=0,
        space_after=5
    )


    story = []


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    story.append(
        safe_para(
            "แผนการจัดการเรียนรู้",
            title
        )
    )


    story.append(
        safe_para(
            f"เรื่อง {summary['topic']}",
            subtitle
        )
    )


    story.append(
        safe_para(

            f"วิชา {summary['subject']}    |    "
            f"ระดับชั้น {summary['grade']}    |    "
            f"เวลา {summary['duration']}    |    "
            f"ครูผู้สอน {teacher}",

            meta
        )
    )


    story.append(
        Spacer(
            1,
            4 * mm
        )
    )


    # =====================================================
    # 1. OBJECTIVE
    # =====================================================

    story.append(
        safe_para(
            "1. จุดประสงค์การเรียนรู้",
            h1
        )
    )


    for x in lesson.get(
        "objective",
        []
    ):

        story.append(
            safe_para(
                f"- {x}",
                bullet
            )
        )


    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    # =====================================================
    # 2. CONTENT
    # =====================================================

    story.append(
        safe_para(
            "2. เนื้อหาที่ใช้สอน",
            h1
        )
    )


    content = (
        lesson.get("content")
        or lesson.get("material")
        or []
    )


    if isinstance(
        content,
        str
    ):

        content = [content]


    for x in content:

        story.append(
            safe_para(
                x,
                body
            )
        )


    # =====================================================
    # KEY POINT
    # =====================================================

    story.append(
        safe_para(
            "สาระสำคัญ",
            h2
        )
    )


    key = (
        lesson.get("key_points")
        or lesson.get("important")
        or []
    )


    if isinstance(
        key,
        str
    ):

        key = [key]


    for x in key:

        story.append(
            safe_para(
                x,
                body
            )
        )


    # =====================================================
    # EXAMPLES
    # =====================================================

    examples = (
        lesson.get("examples")
        or []
    )


    if examples:

        story.append(
            safe_para(
                "ตัวอย่างสำหรับใช้สอน",
                h2
            )
        )


        if isinstance(
            examples,
            str
        ):

            examples = [examples]


        for x in examples:

            story.append(
                safe_para(
                    x,
                    body
                )
            )


    # =====================================================
    # STEPS
    # =====================================================

    story.append(
        safe_para(
            "3. ขั้นตอนการจัดการเรียนรู้",
            h1
        )
    )


    for step in lesson.get(
        "steps",
        []
    ):

        time = clean_text(
            step.get("time")
        )

        title_text = clean_text(
            step.get("title")
        )

        detail = clean_text(
            step.get("detail")
        )


        story.append(
            safe_para(
                f"{time}  {title_text}",
                h2
            )
        )


        story.append(
            safe_para(
                detail,
                body
            )
        )


    # =====================================================
    # ASSESSMENT
    # =====================================================

    story.append(
        safe_para(
            "4. การประเมินผล",
            h1
        )
    )


    story.append(
        safe_para(
            lesson.get(
                "assessment",
                ""
            ),
            body
        )
    )


    doc.build(
        story
    )


# =========================================================
# WORKSHEET PDF
# =========================================================

def build_worksheet_pdf(
    data: dict,
    path: Path
):

    if FONT_ERROR:

        raise RuntimeError(
            FONT_ERROR
        )


    summary = data["summary"]

    teacher = (
        summary.get("teacher")
        or "ไม่ระบุ"
    )


    doc = SimpleDocTemplate(

        str(path),

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"ใบงาน - "
            f"{summary['topic']}"
        )
    )


    title = pstyle(
        "WT",
        25,
        30,
        True,
        TA_CENTER,
        space_after=2
    )


    sub = pstyle(
        "WS",
        18,
        23,
        False,
        TA_CENTER,
        space_after=13
    )


    meta = pstyle(
        "WM",
        16,
        21,
        False,
        TA_CENTER,
        space_after=14
    )


    normal = pstyle(
        "WN",
        16,
        23,
        False,
        TA_LEFT,
        space_after=7
    )


    question = pstyle(
        "WQ",
        17,
        25,
        False,
        TA_LEFT,
        first=0,
        space_after=6
    )


    option = pstyle(
        "WO",
        16,
        23,
        False,
        TA_LEFT,
        first=10,
        space_after=3
    )


    bold = pstyle(
        "WB",
        17,
        24,
        True,
        TA_LEFT,
        space_before=5,
        space_after=7
    )


    story = []


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    story.append(
        safe_para(
            "ใบงาน",
            title
        )
    )


    story.append(
        safe_para(
            f"เรื่อง {summary['topic']}",
            sub
        )
    )


    story.append(
        safe_para(

            f"วิชา {summary['subject']}    |    "
            f"ระดับชั้น {summary['grade']}    |    "
            f"เวลา {summary['duration']}    |    "
            f"ครูผู้สอน {teacher}",

            meta
        )
    )


    story.append(
        safe_para(
            "ชื่อ-สกุล ................................................................................................",
            normal
        )
    )


    story.append(
        safe_para(
            "ชั้น .................. เลขที่ .................. วันที่ ..................",
            normal
        )
    )


    story.append(
        Spacer(
            1,
            3 * mm
        )
    )


    story.append(
        safe_para(
            "คำชี้แจง",
            bold
        )
    )


    story.append(
        safe_para(
            "ให้นักเรียนอ่านคำถามแต่ละข้อ และเขียนคำตอบลงในพื้นที่ที่กำหนด",
            normal
        )
    )


    story.append(
        Spacer(
            1,
            2 * mm
        )
    )


    # =====================================================
    # QUESTIONS
    # =====================================================

    letters = [
        "ก.",
        "ข.",
        "ค.",
        "ง.",
        "จ."
    ]


    for i, q in enumerate(
        data.get("worksheet", []),
        1
    ):

        # -------------------------------------------------
        # NUMBER + QUESTION
        # -------------------------------------------------

        question_text = (
            f"{i}. "
            f"{q.get('question', '')}"
        )


        story.append(
            safe_para(
                question_text,
                question
            )
        )


        # -------------------------------------------------
        # OPTIONS
        # -------------------------------------------------

        options = (
            q.get("options")
            or []
        )


        for j, op in enumerate(
            options
        ):

            if j < len(letters):

                story.append(
                    safe_para(
                        f"{letters[j]} {op}",
                        option
                    )
                )


        # -------------------------------------------------
        # ANSWER LINE
        # -------------------------------------------------

        story.append(
            safe_para(

                "คำตอบ ................................................................................................................",

                normal
            )
        )


        story.append(
            Spacer(
                1,
                5 * mm
            )
        )


    doc.build(
        story
    )


# =========================================================
# QUIZ PDF
# =========================================================

def build_quiz_pdf(
    data: dict,
    path: Path
):

    if FONT_ERROR:

        raise RuntimeError(
            FONT_ERROR
        )


    summary = data["summary"]

    teacher = (
        summary.get("teacher")
        or "ไม่ระบุ"
    )


    doc = SimpleDocTemplate(

        str(path),

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"แบบทดสอบ - "
            f"{summary['topic']}"
        )
    )


    title = pstyle(
        "QT",
        25,
        30,
        True,
        TA_CENTER,
        space_after=2
    )


    sub = pstyle(
        "QS",
        18,
        23,
        False,
        TA_CENTER,
        space_after=13
    )


    meta = pstyle(
        "QM",
        16,
        21,
        False,
        TA_CENTER,
        space_after=14
    )


    normal = pstyle(
        "QN",
        16,
        23,
        False,
        TA_LEFT,
        space_after=6
    )


    qstyle = pstyle(
        "QQ",
        17,
        25,
        False,
        TA_LEFT,
        space_after=6
    )


    opt = pstyle(
        "QO",
        16,
        23,
        False,
        TA_LEFT,
        first=10,
        space_after=3
    )


    story = []


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    story.append(
        safe_para(
            "แบบทดสอบ",
            title
        )
    )


    story.append(
        safe_para(
            f"เรื่อง {summary['topic']}",
            sub
        )
    )


    story.append(
        safe_para(

            f"วิชา {summary['subject']}    |    "
            f"ระดับชั้น {summary['grade']}    |    "
            f"เวลา {summary['duration']}    |    "
            f"ครูผู้สอน {teacher}",

            meta
        )
    )


    story.append(
        safe_para(
            "ชื่อ-สกุล ................................................................................................",
            normal
        )
    )


    story.append(
        safe_para(
            "ชั้น .................. เลขที่ ..................",
            normal
        )
    )


    story.append(
        Spacer(
            1,
            3 * mm
        )
    )


    story.append(
        safe_para(
            "คำชี้แจง ให้นักเรียนทำแบบทดสอบทุกข้อ และเลือกหรือเขียนคำตอบให้ถูกต้อง",
            normal
        )
    )


    story.append(
        Spacer(
            1,
            3 * mm
        )
    )


    letters = [
        "ก.",
        "ข.",
        "ค.",
        "ง.",
        "จ."
    ]


    # =====================================================
    # QUESTIONS
    # =====================================================

    for i, q in enumerate(
        data.get("quiz", []),
        1
    ):

        # IMPORTANT:
        # ไม่แสดง (ปรนัย)
        # ไม่แสดงชื่อประเภทข้อสอบ

        story.append(
            safe_para(
                f"{i}. {q.get('question', '')}",
                qstyle
            )
        )


        options = (
            q.get("options")
            or []
        )


        for j, op in enumerate(
            options
        ):

            if j < len(letters):

                story.append(
                    safe_para(
                        f"{letters[j]} {op}",
                        opt
                    )
                )


        story.append(
            Spacer(
                1,
                5 * mm
            )
        )


    doc.build(
        story
    )


# =========================================================
# ANSWER PDF
# =========================================================

def build_answer_pdf(
    data: dict,
    path: Path
):

    if FONT_ERROR:

        raise RuntimeError(
            FONT_ERROR
        )


    summary = data["summary"]

    teacher = (
        summary.get("teacher")
        or "ไม่ระบุ"
    )


    doc = SimpleDocTemplate(

        str(path),

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"เฉลย - "
            f"{summary['topic']}"
        )
    )


    title = pstyle(
        "AT",
        25,
        30,
        True,
        TA_CENTER,
        space_after=2
    )


    sub = pstyle(
        "AS",
        18,
        23,
        False,
        TA_CENTER,
        space_after=13
    )


    meta = pstyle(
        "AM",
        16,
        21,
        False,
        TA_CENTER,
        space_after=14
    )


    h = pstyle(
        "AH",
        19,
        24,
        True,
        TA_LEFT,
        space_before=7,
        space_after=8
    )


    q = pstyle(
        "AQ",
        16,
        23,
        False,
        TA_LEFT,
        space_after=7
    )


    story = []


    story.append(
        safe_para(
            "เฉลย",
            title
        )
    )


    story.append(
        safe_para(
            f"เรื่อง {summary['topic']}",
            sub
        )
    )


    story.append(
        safe_para(

            f"วิชา {summary['subject']}    |    "
            f"ระดับชั้น {summary['grade']}    |    "
            f"เวลา {summary['duration']}    |    "
            f"ครูผู้สอน {teacher}",

            meta
        )
    )


    # =====================================================
    # WORKSHEET ANSWERS
    # =====================================================

    story.append(
        safe_para(
            "เฉลยใบงาน",
            h
        )
    )


    for i, x in enumerate(
        data.get("worksheet", []),
        1
    ):

        story.append(
            safe_para(
                f"{i}. {x.get('answer', '')}",
                q
            )
        )


    # =====================================================
    # QUIZ ANSWERS
    # =====================================================

    story.append(
        safe_para(
            "เฉลยแบบทดสอบ",
            h
        )
    )


    for i, x in enumerate(
        data.get("quiz", []),
        1
    ):

        answer = x.get(
            "answer",
            ""
        )

        explanation = x.get(
            "explanation",
            ""
        )


        story.append(
            safe_para(
                f"{i}. {answer}",
                q
            )
        )


        if explanation:

            story.append(
                safe_para(
                    explanation,
                    q
                )
            )


    doc.build(
        story
    )


# =========================================================
# CREATE ALL PDF
# =========================================================

def create_pdfs(
    data: dict
):

    uid = uuid.uuid4().hex


    files = {

        "lesson_pdf":
            PDF_DIR /
            f"{uid}_lesson.pdf",

        "worksheet_pdf":
            PDF_DIR /
            f"{uid}_worksheet.pdf",

        "quiz_pdf":
            PDF_DIR /
            f"{uid}_quiz.pdf",

        "answer_pdf":
            PDF_DIR /
            f"{uid}_answer.pdf",

    }


    build_lesson_pdf(
        data,
        files["lesson_pdf"]
    )


    build_worksheet_pdf(
        data,
        files["worksheet_pdf"]
    )


    build_quiz_pdf(
        data,
        files["quiz_pdf"]
    )


    build_answer_pdf(
        data,
        files["answer_pdf"]
    )


    return {

        key:
            f"/api/pdf/{value.name}"

        for key, value
        in files.items()

    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "version": "1.5",

        "font_regular":
            FONT_REGULAR.name,

        "font_bold":
            FONT_BOLD.name,

        "font_error":
            FONT_ERROR,

        "openai_configured":
            bool(
                os.getenv(
                    "OPENAI_API_KEY"
                )
            ),

    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def index():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# =========================================================
# PDF DOWNLOAD
# =========================================================

@app.get(
    "/api/pdf/{filename}"
)
def get_pdf(
    filename: str
):

    safe = Path(
        filename
    ).name


    path = (
        PDF_DIR /
        safe
    )


    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์ PDF"
        )


    return FileResponse(

        path,

        media_type="application/pdf",

        filename=safe

    )


# =========================================================
# GENERATE
# =========================================================

@app.post(
    "/api/generate"
)
def generate(
    req: GenerateRequest
):

    client = get_openai()


    system = """
คุณเป็นผู้ช่วยจัดทำเอกสารการเรียนการสอนภาษาไทยสำหรับครู

สร้างข้อมูลเป็น JSON เท่านั้น

ห้ามใส่ Markdown

ห้ามใส่ข้อความนอก JSON


รูปแบบ JSON:

{
  "summary": {
    "grade": "",
    "subject": "",
    "topic": "",
    "duration": ""
  },

  "lesson_plan": {
    "objective": [],
    "content": [],
    "key_points": [],
    "examples": [],
    "steps": [
      {
        "time": "",
        "title": "",
        "detail": ""
      }
    ],
    "assessment": ""
  },

  "worksheet": [
    {
      "question": "",
      "answer": "",
      "options": []
    }
  ],

  "quiz": [
    {
      "question": "",
      "answer": "",
      "explanation": "",
      "options": [],
      "type": ""
    }
  ]
}


กติกาสำคัญ:

1. ใช้ภาษาไทยเป็นหลัก
2. เว้นแต่หัวข้อเป็นภาษาอื่น
3. ห้ามใส่คำว่า "AI ครูผู้ช่วย" ในเอกสาร
4. ห้ามใส่ชื่อประเภทข้อสอบลงหน้าคำถาม
5. ห้ามเขียน "(ปรนัย)"
6. ห้ามเขียน "(เติมคำ)"
7. ห้ามเขียน "(คำนวณ)"
8. ห้ามเขียน "(ประยุกต์ใช้)"
9. question ต้องมีเฉพาะตัวคำถาม
10. ห้ามใส่เลขข้อใน question
11. ระบบจะใส่เลขข้อเอง
12. ถ้าเป็นปรนัย ให้ใส่ตัวเลือกใน options เป็นข้อความล้วน
13. ไม่ต้องใส่ ก. ข. ค. ง. ใน options
14. ใบงานควรมีพื้นที่ให้ตอบ
15. แบบทดสอบควรมีคำถามชัดเจน
16. จำนวนข้อแบบทดสอบต้องตามที่ผู้ใช้กำหนด
17. เนื้อหาควรเหมาะสมกับระดับชั้น
18. เรียบเรียงภาษาให้เหมือนเอกสารการเรียนการสอนจริง
19. หลีกเลี่ยงข้อความที่เหมือนการก๊อปปี้วาง
20. แต่ละหัวข้อควรเป็นย่อหน้าอ่านง่าย
"""


    user = {

        "หัวข้อ":
            req.prompt,

        "ชื่อครู":
            req.teacher_name,

        "จำนวนข้อสอบ":
            req.question_count,

        "รูปแบบข้อสอบ":
            req.question_types,

        "ระดับความยาก":
            req.difficulty,

    }


    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=os.getenv(
                    "OPENAI_MODEL",
                    "gpt-4o-mini"
                ),

                response_format={
                    "type":
                    "json_object"
                },

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            system
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                user,
                                ensure_ascii=False
                            )
                    }

                ],

                temperature=0.7,

            )
        )


        raw = (
            response
            .choices[0]
            .message
            .content
        )


        data = json.loads(
            raw
        )


        data = normalize_data(
            data,
            req
        )


        data["pdf"] = create_pdfs(
            data
        )


        return JSONResponse(
            data
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# =========================================================
# STATIC
# =========================================================

app.mount(

    "/static",

    StaticFiles(
        directory=STATIC_DIR
    ),

    name="static"

)
