from pydantic import BaseModel, Field

class ImageIndex(BaseModel):
    xref: int
    local_path: str

class AgentDependencies(BaseModel):
    full_text: str
    image_map: dict[int, str]

class PaperAnalysisReport(BaseModel):
    title: str = Field(..., description="The title of the paper.")
    authors: list[str] = Field(..., description="The authors of the paper.")
    abstract: str = Field(..., description="The abstract of the paper.")
    introduction: str = Field(..., description="The introduction of the paper.")
    conclusion: str = Field(..., description="The conclusion of the paper.")
    body: str = Field(..., description="The main body of the paper.")
