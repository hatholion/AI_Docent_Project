const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type ClassifyCandidate = {
  artifact_id: string;
  confidence: number;
};

export type ClassifyResponse = {
  artifact_id: string | null;
  artifact_name: string | null;
  confidence: number;
  candidates: ClassifyCandidate[];
  message?: string;
};

export async function classifyImage(image: Blob): Promise<ClassifyResponse> {
  const formData = new FormData();
  formData.append("image", image, "frame.jpg");

  const response = await fetch(`${API_BASE_URL}/api/v1/classify`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`분류 요청 실패 (status ${response.status})`);
  }
  return response.json();
}
