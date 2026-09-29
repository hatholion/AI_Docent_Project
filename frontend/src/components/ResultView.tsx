import { useCallback, useEffect, useState } from "react";
import { Button, Mascot, artifactPhoto } from "./common";
import { AssistantMessage, ChatPanel, errorMessage, isAbort } from "./ChatPanel";
import {
  fetchArtifact,
  fetchDescription,
  type ArtifactDetail,
  type DescriptionResponse,
} from "../api/docent";
import type { RecognizedArtifact, VisitorType } from "../types";

type LoadState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; message: string };

const VISITOR_LABELS: Record<VisitorType, string> = {
  child: "어린이 눈높이",
  general: "일반인 눈높이",
  expert: "전문가 눈높이",
};

// 인식 직후 한 번 불러오고, retry를 호출하면 다시 불러온다
function useLoad<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
): [LoadState<T>, () => void] {
  const [state, setState] = useState<LoadState<T>>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });
    load(controller.signal)
      .then((data) => setState({ status: "ready", data }))
      .catch((error) => {
        if (isAbort(error)) return;
        setState({ status: "error", message: errorMessage(error) });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  return [state, () => setAttempt((count) => count + 1)];
}

function FactValue({
  state,
  pick,
}: {
  state: LoadState<ArtifactDetail>;
  pick: (detail: ArtifactDetail) => string | null;
}) {
  if (state.status === "loading") return <dd className="fact-loading">불러오는 중</dd>;
  if (state.status === "error") return <dd className="fact-missing">-</dd>;
  const value = pick(state.data);
  return <dd>{value ?? <span className="fact-missing">정보 없음</span>}</dd>;
}

export function ResultView({
  artifact,
  visitorType,
  imageUrl = artifactPhoto,
  onRescan,
}: {
  artifact: RecognizedArtifact;
  visitorType: VisitorType;
  imageUrl?: string;
  onRescan: () => void;
}) {
  const { artifact_id: artifactId } = artifact;
  const [chatOpen, setChatOpen] = useState(false);
  const closeChat = useCallback(() => setChatOpen(false), []);

  const [detail, retryDetail] = useLoad(
    (signal) => fetchArtifact(artifactId, signal),
    [artifactId],
  );
  const [description, retryDescription] = useLoad<DescriptionResponse>(
    (signal) =>
      fetchDescription({ artifact_id: artifactId, visitor_type: visitorType }, signal),
    [artifactId, visitorType],
  );

  const artifactName =
    detail.status === "ready" ? detail.data.artifact_name : artifact.artifact_name;
  const confidencePercent = Math.round(artifact.confidence * 100);
  const source = detail.status === "ready" ? detail.data.source : null;

  return (
    <div className="result-view">
      <aside className="artifact-panel">
        <div className="frozen-photo">
          <img src={imageUrl} alt={`인식된 ${artifactName}`} />
        </div>
        <Button className="secondary-button" onClick={onRescan}>
          다시 비추기
        </Button>
      </aside>

      <section className="info-panel" aria-label="유물 정보">
        <div className="info-heading">
          <h2>{artifactName}</h2>
          <p className="info-subtitle">
            {source ?? "국립중앙박물관"}
            <span className="confidence-note"> · AI 인식 신뢰도 {confidencePercent}%</span>
          </p>
        </div>

        <article className="commentary">
          <div className="commentary-heading">
            <span className="section-kicker">민속이의 해설</span>
            <span className="visitor-chip">{VISITOR_LABELS[visitorType]}</span>
          </div>
          {description.status === "loading" && <AssistantMessage waiting />}
          {description.status === "ready" && (
            <>
              <p className="commentary-body">{description.data.description}</p>
              {description.data.sources.length > 0 && (
                <span className="source">출처: {description.data.sources.join(", ")}</span>
              )}
            </>
          )}
          {description.status === "error" && (
            <p className="info-error">
              해설을 불러오지 못했어요. {description.message}
              <Button className="retry-button" onClick={retryDescription}>
                다시 시도
              </Button>
            </p>
          )}
        </article>

        <div className="detail-section">
          <span className="section-kicker">상세 정보</span>
          <dl className="fact-list">
            <div>
              <dt>소장품 번호</dt>
              <FactValue state={detail} pick={(d) => d.accession_no} />
            </div>
            <div>
              <dt>지정 구분</dt>
              <FactValue state={detail} pick={(d) => d.designation_no} />
            </div>
            <div>
              <dt>소장처</dt>
              <FactValue state={detail} pick={(d) => d.source} />
            </div>
          </dl>
          {detail.status === "error" && (
            <p className="info-error">
              기본 정보를 불러오지 못했어요. {detail.message}
              <Button className="retry-button" onClick={retryDetail}>
                다시 시도
              </Button>
            </p>
          )}
        </div>
      </section>

      {!chatOpen && (
        <button
          type="button"
          className="chat-fab"
          onClick={() => setChatOpen(true)}
        >
          <Mascot />
          <span className="chat-fab-label">민속이에게 물어보기</span>
        </button>
      )}

      <ChatPanel
        artifactId={artifactId}
        artifactName={artifactName}
        visitorType={visitorType}
        open={chatOpen}
        onClose={closeChat}
      />
    </div>
  );
}
