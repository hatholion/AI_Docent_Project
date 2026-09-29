// docs/api_spec.md 2·3·4·5번 엔드포인트 클라이언트.
// 스키마는 팀 합의 사항이므로 변경 시 api_spec.md와 함께 수정한다.
import type { VisitorType } from "../types";
import * as mock from "./mockDocent";

export type ArtifactDetail = {
  artifact_id: string;
  accession_no: string | null;
  artifact_name: string;
  source: string;
  description: string;
  designation_no: string | null;
};

export type DescriptionRequest = {
  artifact_id: string;
  visitor_type: VisitorType;
};

export type DescriptionResponse = {
  artifact_id: string;
  artifact_name: string;
  visitor_type: VisitorType;
  description: string;
  sources: string[];
};

export type ChatRequest = {
  artifact_id: string;
  visitor_type: VisitorType;
  session_id?: string;
  question: string;
};

export type ChatResponse = {
  session_id: string;
  artifact_id: string;
  visitor_type: VisitorType;
  answer: string;
  sources: string[];
};

export type ChatHistoryResponse = {
  session_id: string;
  artifact_id: string;
  visitor_type: VisitorType;
  messages: { role: "user" | "assistant"; content: string }[];
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

// 백엔드 구현 전까지는 mock 응답을 사용한다. 실제 서버 연결 시 VITE_USE_MOCK=false.
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";
// lib/api.ts(분류 API)와 같은 서버 주소를 쓴다
const API_BASE = `${import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"}/api/v1`;

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(0, "NETWORK_ERROR", "서버에 연결할 수 없습니다.");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(
      response.status,
      body?.error?.code ?? "UNKNOWN_ERROR",
      body?.error?.message ?? "요청을 처리하지 못했습니다.",
    );
  }
  return body as T;
}

// GET /artifacts/{artifact_id} 는 인식 담당 엔드포인트. 결과 화면 정보 카드에서 사용한다.
export function fetchArtifact(
  artifactId: string,
  signal?: AbortSignal,
): Promise<ArtifactDetail> {
  if (USE_MOCK) return mock.fetchArtifact(artifactId, signal);
  return request(`/artifacts/${encodeURIComponent(artifactId)}`, {
    method: "GET",
    signal,
  });
}

export function fetchDescription(
  payload: DescriptionRequest,
  signal?: AbortSignal,
): Promise<DescriptionResponse> {
  if (USE_MOCK) return mock.fetchDescription(payload, signal);
  return request("/docent/description", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export function sendChat(
  payload: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  if (USE_MOCK) return mock.sendChat(payload, signal);
  return request("/docent/chat", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export function fetchChatHistory(
  sessionId: string,
  signal?: AbortSignal,
): Promise<ChatHistoryResponse> {
  if (USE_MOCK) return mock.fetchChatHistory(sessionId, signal);
  return request(`/docent/chat/${encodeURIComponent(sessionId)}`, {
    method: "GET",
    signal,
  });
}
