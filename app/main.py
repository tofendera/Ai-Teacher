import os
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
    version="1.8"
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
# HELPERS
# =========================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def normalize_list(value):

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return [value]

    return [str(value)]


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


    # =====================================================
    # LESSON
    # =====================================================

    lesson = data.get("lesson_plan") or {}

    lesson["objective"] = normalize_list(
        lesson.get("objective")
    )

    lesson["content"] = normalize_list(
        lesson.get("content")
    )

    lesson["key_points"] = normalize_list(
        lesson.get("key_points")
    )

    lesson["examples"] = normalize_list(
        lesson.get("examples")
    )

    lesson["steps"] = (
        lesson.get("steps")
        or []
    )

    lesson["assessment"] = clean_text(
        lesson.get("assessment")
    )


    # =====================================================
    # WORKSHEET
    # =====================================================

    worksheet = data.get("worksheet") or []

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


    # =====================================================
    # QUIZ
    # =====================================================

    quiz = data.get("quiz") or []

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


def add_paragraphs(
    story,
    values,
    style
):

    values = normalize_list(values)

    for value in values:

        if isinstance(value, dict):

            title = clean_text(
                value.get("title")
            )

            explanation = clean_text(
                value.get("explanation")
            )

            if title:

                story.append(
                    safe_para(
                        title,
                        style
                    )
                )

            if explanation:

                story.append(
                    safe_para(
                        explanation,
                        style
                    )
                )

        else:

            text = clean_text(value)

            if text:

                # รองรับข้อความที่มีหลายย่อหน้า
                paragraphs = [
                    x.strip()
                    for x in text.split("\n\n")
                    if x.strip()
                ]

                for paragraph in paragraphs:

                    story.append(
                        safe_para(
                            paragraph,
                            style
                        )
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
        27,
        32,
        True,
        TA_CENTER,
        space_after=3
    )


    subtitle = pstyle(
        "ST",
        19,
        24,
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
        space_after=16
    )


    h1 = pstyle(
        "H1",
        20,
        26,
        True,
        TA_LEFT,
        space_before=10,
        space_after=9
    )


    h2 = pstyle(
        "H2",
        18,
        24,
        True,
        TA_LEFT,
        space_before=9,
        space_after=7
    )


    body = pstyle(
        "B",
        17,
        25,
        False,
        TA_LEFT,
        first=10,
        space_after=9
    )


    bullet = pstyle(
        "BL",
        17,
        25,
        False,
        TA_LEFT,
        first=0,
        space_after=5
    )


    story = []


    # =====================================================
    # HEADER
    # =====================================================

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


    # =====================================================
    # 1 OBJECTIVE
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
                f"- {clean_text(x)}",
                bullet
            )
        )


    # =====================================================
    # 2 CONTENT
    # =====================================================

    story.append(
        safe_para(
            "2. เนื้อหาที่ใช้สอน",
            h1
        )
    )


    add_paragraphs(
        story,
        lesson.get("content"),
        body
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


    add_paragraphs(
        story,
        lesson.get("key_points"),
        body
    )


    # =====================================================
    # EXAMPLES
    # =====================================================

    examples = lesson.get(
        "examples",
        []
    )


    if examples:

        story.append(
            safe_para(
                "ตัวอย่างสำหรับใช้สอน",
                h2
            )
        )


        add_paragraphs(
            story,
            examples,
            body
        )


    # =====================================================
    # 3 LESSON STEPS
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


        add_paragraphs(
            story,
            [detail],
            body
        )


    # =====================================================
    # 4 ASSESSMENT
    # =====================================================

    story.append(
        safe_para(
            "4. การประเมินผล",
            h1
        )
    )


    add_paragraphs(
        story,
        [lesson.get("assessment", "")],
        body
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
        27,
        32,
        True,
        TA_CENTER,
        space_after=3
    )


    sub = pstyle(
        "WS",
        19,
        24,
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


    student = pstyle(
        "WST",
        16,
        21,
        False,
        TA_LEFT,
        space_after=8
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
        18,
        24,
        True,
        TA_LEFT,
        space_before=8,
        space_after=7
    )


    story = []


    # =====================================================
    # HEADER
    # =====================================================

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


    # =====================================================
    # STUDENT INFO — ONE LINE
    # =====================================================

    story.append(
        safe_para(

            "ชื่อ-สกุล .........................................................    "
            "ชั้น ...............    "
            "เลขที่ ...............",

            student
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
        27,
        32,
        True,
        TA_CENTER,
        space_after=3
    )


    sub = pstyle(
        "QS",
        19,
        24,
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


    student = pstyle(
        "QST",
        16,
        21,
        False,
        TA_LEFT,
        space_after=8
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


    # =====================================================
    # HEADER
    # =====================================================

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


    # =====================================================
    # STUDENT INFO — ONE LINE
    # =====================================================

    story.append(
        safe_para(

            "ชื่อ-สกุล .........................................................    "
            "ชั้น ...............    "
            "เลขที่ ...............",

            student
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
        27,
        32,
        True,
        TA_CENTER,
        space_after=3
    )


    sub = pstyle(
        "AS",
        19,
        24,
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
        20,
        26,
        True,
        TA_LEFT,
        space_before=10,
        space_after=9
    )


    q = pstyle(
        "AQ",
        17,
        24,
        False,
        TA_LEFT,
        space_after=7
    )


    story = []


    # =====================================================
    # HEADER
    # =====================================================

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

        "version": "1.8",

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


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    system = """
คุณเป็นผู้ช่วยจัดทำเอกสารการเรียนการสอนภาษาไทยสำหรับครูไทย

หน้าที่ของคุณคือสร้าง "ชุดการสอน" ที่สามารถนำไปใช้สอนได้จริง
ไม่ใช่เพียงสร้างรายการหัวข้อสั้น ๆ

ตอบเป็น JSON เท่านั้น

ห้ามใส่ Markdown

ห้ามใส่ข้อความนอก JSON


=========================================================
โครงสร้าง JSON
=========================================================

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


=========================================================
กฎสำคัญเกี่ยวกับแผนการสอน
=========================================================

1. ต้องสร้างเนื้อหาที่สามารถนำไปใช้สอนได้จริง

2. ห้ามตอบเพียงชื่อหัวข้อหรือคำสั้น ๆ

3. "objective" ต้องเป็นจุดประสงค์การเรียนรู้ที่ชัดเจน
   อย่างน้อย 3 ข้อ

4. "content" ต้องเป็นเนื้อหาความรู้จริง
   สำหรับให้ครูใช้สอนนักเรียน
   ต้องอธิบายแนวคิด ความหมาย หลักการ หรือวิธีการ
   ตามหัวข้อที่ผู้ใช้ระบุ

5. "key_points" คือ "สาระสำคัญ"
   ต้องเป็นข้อความอธิบายเนื้อหาจริง
   ไม่ใช่เพียงคำสำคัญหรือรายการคำศัพท์

6. สาระสำคัญควรสรุปว่า
   นักเรียนควรเข้าใจอะไร
   หลักการสำคัญคืออะไร
   และสามารถนำความรู้นั้นไปใช้อย่างไร

7. content และ key_points ต้องมีรายละเอียดเพียงพอ
   ไม่ใช่ข้อความ 1 บรรทัดสั้น ๆ

8. ถ้าเป็นเนื้อหาที่อธิบายได้หลายประเด็น
   ให้แยกเป็นหลายย่อหน้าหรือหลายรายการ

9. "examples" ต้องมีตัวอย่างที่ครูสามารถหยิบไปใช้สอนได้จริง

10. ตัวอย่างควรอธิบายพร้อมตัวอย่างประกอบ
    ไม่ใช่เพียงตั้งชื่อหัวข้อ

11. "steps" ต้องอธิบายขั้นตอนการสอนจริง

12. แต่ละ step ต้องมี
    - เวลา
    - ชื่อกิจกรรม
    - รายละเอียดว่าครูทำอะไรและนักเรียนทำอะไร

13. "assessment" ต้องบอกวิธีประเมินจริง
    เช่น การสังเกต การตอบคำถาม การทำใบงาน
    หรือการตรวจแบบทดสอบ

14. เนื้อหาทั้งหมดต้องเหมาะสมกับระดับชั้น

15. หากผู้ใช้ระบุเวลา เช่น 30 นาที หรือ 1 ชั่วโมง
    ให้จัดกิจกรรมให้สอดคล้องกับเวลานั้น


=========================================================
รูปแบบภาษา
=========================================================

1. ใช้ภาษาไทยเป็นหลัก
2. ใช้ภาษาที่เหมาะกับเอกสารการเรียนการสอนจริง
3. เขียนให้ครูสามารถนำไปใช้ได้ทันที
4. หลีกเลี่ยงข้อความที่ดูเหมือน AI เขียนแบบสั้น ๆ
5. ไม่เขียนซ้ำความหมายเดิมหลายครั้ง
6. เรียบเรียงเป็นธรรมชาติ
7. หากเนื้อหามีหลายแนวคิด ให้แยกเป็นย่อหน้า
8. ห้ามใช้ Emoji ในข้อมูลที่จะนำไปสร้าง PDF


=========================================================
กฎเกี่ยวกับข้อสอบ
=========================================================

1. จำนวนข้อสอบต้องตรงกับจำนวนที่ผู้ใช้กำหนด

2. question ต้องมีเฉพาะข้อความคำถาม

3. ห้ามใส่เลขข้อใน question

4. ห้ามใส่ประเภทข้อสอบใน question

5. ห้ามเขียน "(ปรนัย)"

6. ห้ามเขียน "(เติมคำ)"

7. ห้ามเขียน "(คำนวณ)"

8. ห้ามเขียน "(ประยุกต์ใช้)"

9. ถ้าเป็นปรนัย ให้ใส่ตัวเลือกใน options

10. ไม่ต้องใส่ ก. ข. ค. ง. ใน options

11. ถ้าเป็นปรนัยต้องมี 4 ตัวเลือก

12. มีคำตอบถูกเพียง 1 ตัวเลือก

13. explanation ต้องอธิบายเหตุผลของคำตอบ

14. คำถามต้องสอดคล้องกับเนื้อหาที่สอน


=========================================================
กฎเกี่ยวกับใบงาน
=========================================================

1. ใบงานต้องสอดคล้องกับเนื้อหาที่สอน

2. question ต้องไม่มีเลขข้อ

3. ระบบจะใส่เลขข้อเอง

4. ถ้าเป็นคำถามปลายเปิด ให้ answer เป็นแนวคำตอบ

5. ถ้าเป็นปรนัย ให้ใส่ options

6. ไม่ต้องใส่ ก. ข. ค. ง. ใน options

7. ใบงานควรมีทั้งคำถามความเข้าใจ
   และคำถามที่ให้นักเรียนคิดหรือประยุกต์ใช้ตามความเหมาะสม


=========================================================
กฎชื่อครู
=========================================================

ห้ามสร้างชื่อครูขึ้นมาเอง

ชื่อครูต้องใช้จากข้อมูลที่ผู้ใช้ส่งมาเท่านั้น

ถ้าไม่ได้ระบุชื่อครู ให้ส่งค่าเป็นค่าว่าง


=========================================================
กฎสำคัญเพิ่มเติม
=========================================================

ห้ามใส่คำว่า "AI ครูผู้ช่วย"

ห้ามใส่คำว่า "AI-Teacher"

ห้ามใส่ข้อความโฆษณา

ห้ามใส่คำอธิบายเกี่ยวกับการทำงานของ AI

ห้ามสร้างหัวข้อเพิ่มที่ไม่มีในโครงสร้าง JSON

ตอบ JSON เท่านั้น
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
