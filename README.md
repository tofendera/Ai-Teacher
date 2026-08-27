# AI ครูผู้ช่วย V1

Prototype: พิมพ์คำสั่งเดียว เช่น "เศษส่วน ป.4 1 ชั่วโมง"
แล้ว AI สร้าง แผนการสอน + ใบงาน + เฉลย + แบบทดสอบ

## รันบนเครื่อง
1. ติดตั้ง Python 3.12+
2. `pip install -r requirements.txt`
3. ตั้ง `OPENAI_API_KEY`
4. `uvicorn app.main:app --reload`
5. เปิด http://127.0.0.1:8000

## Deploy บน Render
สร้าง Web Service จาก GitHub repository นี้
- Runtime: Docker
- Environment Variable: `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL`

หมายเหตุ: อย่าใส่ API key ลงในโค้ดหรือ GitHub
