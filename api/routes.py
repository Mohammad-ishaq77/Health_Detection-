from fastapi import APIRouter
from src.schemas.disease_schema import DiseaseRequest
from src.predict.predict_disease import predict_disease
from src.schemas.heart_schema import HeartRequest
from src.predict.predict_heart import predict_heart
from src.schemas.diabetes_schema import DiabetesRequest
from src.predict.predict_diabetes import predict_diabetes


router = APIRouter()

@router.get("/")
def home():
    return {"message": "AI Health ML Service Running 🚀"}

@router.post("/predict-disease")
def predict(req: DiseaseRequest):
    result = predict_disease(req.symptoms)

    if "error" in result:
        return {
            "status": "error",
            "message": result["error"]
        }

    return {
        "status": "success",
        "data": result
    }


@router.post("/predict-heart-risk")
def predict_heart_api(req: HeartRequest):
    result = predict_heart(req.dict())

    return {
        "status": "success",
        "data": result
    }

@router.post("/predict-diabetes")
def predict_diabetes_api(req: DiabetesRequest):
    result = predict_diabetes(req.dict())

    return {
        "status": "success",
        "data": result
    }
