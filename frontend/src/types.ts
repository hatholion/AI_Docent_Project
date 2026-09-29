export type ViewState = "waiting" | "recognizing" | "success" | "not-found";
export type VisitorType = "child" | "general" | "expert";

// POST /api/v1/classify 성공 응답 중 결과 화면(상태 C)에 필요한 부분
export type RecognizedArtifact = {
  artifact_id: string;
  artifact_name: string;
  confidence: number;
};
