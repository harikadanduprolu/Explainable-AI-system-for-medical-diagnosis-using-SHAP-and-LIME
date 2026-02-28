from fastapi import FastAPI

app = FastAPI()
@app.get("/hello")
async def hello():
    return "welcome to the world of AI in healthcare!"