// 백엔드 구현 전 화면 개발용 mock. 응답 형식은 docs/api_spec.md와 동일하게 유지한다.
import type { VisitorType } from "../types";
import {
  ApiError,
  type ArtifactDetail,
  type ChatHistoryResponse,
  type ChatRequest,
  type ChatResponse,
  type DescriptionRequest,
  type DescriptionResponse,
} from "./docent";

// data/metadata.csv 기준. designation_no는 확인된 값만 채움 (docs/TODO.md)
const ARTIFACTS: Record<string, Omit<ArtifactDetail, "artifact_id" | "source" | "description">> = {
  bon002789: { accession_no: "본관 2789", artifact_name: "금동 반가사유상", designation_no: "국보 제1962-1호" },
  duk000798: { accession_no: "덕수 798", artifact_name: "청동촛대", designation_no: null },
  jub002084: { accession_no: "접수 2084", artifact_name: "백자 달항아리", designation_no: null },
  ssu001794: { accession_no: "신수 1794", artifact_name: "농경문 청동기", designation_no: null },
  ssu001846: { accession_no: "신수 1846", artifact_name: "방패형 동기", designation_no: null },
  ssu003094: { accession_no: "신수 3094", artifact_name: "요령식 동검", designation_no: null },
  ssu022891: { accession_no: "신수 22891", artifact_name: "빗살무늬토기", designation_no: null },
};

const DESCRIPTION_TEMPLATES: Record<VisitorType, (name: string) => string> = {
  child: (name) =>
    `이건 ${name}이에요! 아주 오래전 사람들이 정성껏 만든 보물이랍니다. 어떤 모양인지 천천히 살펴볼까요?`,
  general: (name) =>
    `이 유물은 ${name}입니다. 당시 사람들의 생활과 믿음, 뛰어난 제작 기술을 함께 보여주는 중요한 유물이에요. 형태와 세부 장식을 가까이에서 살펴보세요.`,
  expert: (name) =>
    `${name}은(는) 제작 시기의 조형 양식과 기술 수준을 파악할 수 있는 중요한 자료입니다. 재질, 제작 기법, 지정 현황은 공식 설명 데이터 연동 후 제공됩니다.`,
};

const sessions = new Map<string, ChatHistoryResponse>();

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
}

function findArtifactName(artifactId: string): string {
  const name = ARTIFACTS[artifactId]?.artifact_name;
  if (!name) {
    throw new ApiError(
      404,
      "ARTIFACT_NOT_FOUND",
      `artifact_id '${artifactId}'에 해당하는 유물을 찾을 수 없습니다.`,
    );
  }
  return name;
}

export async function fetchArtifact(
  artifactId: string,
  signal?: AbortSignal,
): Promise<ArtifactDetail> {
  await wait(500, signal);
  findArtifactName(artifactId);
  return {
    artifact_id: artifactId,
    source: "국립중앙박물관",
    description: "TODO",
    ...ARTIFACTS[artifactId],
  };
}

export async function fetchDescription(
  { artifact_id, visitor_type }: DescriptionRequest,
  signal?: AbortSignal,
): Promise<DescriptionResponse> {
  await wait(1200, signal);
  const artifact_name = findArtifactName(artifact_id);
  return {
    artifact_id,
    artifact_name,
    visitor_type,
    description: DESCRIPTION_TEMPLATES[visitor_type](artifact_name),
    sources: ["국립중앙박물관"],
  };
}

export async function sendChat(
  { artifact_id, visitor_type, session_id, question }: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  await wait(1000, signal);
  const artifactName = findArtifactName(artifact_id);
  const sessionId =
    session_id && sessions.has(session_id)
      ? session_id
      : `sess_mock_${Date.now().toString(36)}`;
  const answer = `"${question}"에 대한 답변은 RAG·LLM 연동 후 제공될 예정이에요. 지금은 ${artifactName} 화면 확인용 예시 답변입니다.`;

  const session = sessions.get(sessionId) ?? {
    session_id: sessionId,
    artifact_id,
    visitor_type,
    messages: [],
  };
  session.messages.push(
    { role: "user", content: question },
    { role: "assistant", content: answer },
  );
  sessions.set(sessionId, session);

  return {
    session_id: sessionId,
    artifact_id,
    visitor_type,
    answer,
    sources: ["국립중앙박물관"],
  };
}

export async function fetchChatHistory(
  sessionId: string,
  signal?: AbortSignal,
): Promise<ChatHistoryResponse> {
  await wait(300, signal);
  const session = sessions.get(sessionId);
  if (!session) {
    throw new ApiError(404, "SESSION_NOT_FOUND", "대화 세션을 찾을 수 없습니다.");
  }
  return structuredClone(session);
}
