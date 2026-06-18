from pydantic import BaseModel, Field
from typing import List

class DiseaseRequest(BaseModel):
    symptoms: List[str] = Field(
        ...,
        min_items=1,
        example=["itching", "skin_rash", "nodal_skin_eruptions"]
    )