import { useEffect, useState } from "react";
import { AppBar } from "./components/common";
import { ScanView } from "./components/ScanView";
import { ResultView } from "./components/ResultView";
import { SelectTypePage } from "./components/SelectTypePage";
import { useCamera } from "./hooks/useCamera";
import { useArtifactScanner } from "./hooks/useArtifactScanner";
import type { RecognizedArtifact, ViewState, VisitorType } from "./types";

// 카메라로 실제 인식하기 전, 결과 화면 레이아웃을 미리 볼 때 쓰는 예시 유물
const PREVIEW_ARTIFACT: RecognizedArtifact = {
  artifact_id: "bon002789",
  artifact_name: "금동 반가사유상",
  confidence: 0.94,
};

const modeLabels: Record<VisitorType, string> = {
  child: "어린이 모드",
  general: "일반인 모드",
  expert: "전문가 모드",
};

export default function App() {
  const [visitorType, setVisitorType] = useState<VisitorType | null>(null);
  const [viewState, setViewState] = useState<ViewState>("waiting");

  const cameraActive = Boolean(visitorType) && viewState !== "success";
  const { videoElementRef, attachVideo, error: cameraError } = useCamera(cameraActive);
  const scanningActive = cameraActive && !cameraError;
  const scanner = useArtifactScanner(videoElementRef, scanningActive);

  useEffect(() => {
    if (scanner.status === "found") {
      setViewState("recognizing");
      const timer = window.setTimeout(() => setViewState("success"), 700);
      return () => window.clearTimeout(timer);
    }
    if (scanner.status === "delayed") {
      setViewState((current) => (current === "success" ? current : "not-found"));
    }
  }, [scanner.status]);

  const recognizedArtifact: RecognizedArtifact =
    scanner.result?.artifact_id && scanner.result.artifact_name
      ? {
          artifact_id: scanner.result.artifact_id,
          artifact_name: scanner.result.artifact_name,
          confidence: scanner.result.confidence,
        }
      : PREVIEW_ARTIFACT;

  const rescan = () => {
    scanner.reset();
    setViewState("waiting");
  };

  if (!visitorType) {
    return (
      <main className="page">
        <div className="app-frame">
          <SelectTypePage onSelect={setVisitorType} />
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <div className={`app-frame ${viewState === "success" ? "success-frame" : ""}`}>
        <AppBar
          modeLabel={modeLabels[visitorType]}
          onBack={() => setVisitorType(null)}
        />
        <div className="frame-content" key={viewState}>
          {viewState === "success" ? (
            <ResultView
              artifact={recognizedArtifact}
              visitorType={visitorType}
              imageUrl={scanner.capturedPhoto ?? undefined}
              onRescan={rescan}
            />
          ) : (
            <ScanView
              state={viewState}
              onDismiss={rescan}
              attachVideo={attachVideo}
              cameraError={cameraError}
              connectionError={scanner.connectionError}
              onRetry={scanner.retry}
            />
          )}
        </div>
      </div>
    </main>
  );
}
