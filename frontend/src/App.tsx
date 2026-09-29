import { useEffect, useState } from "react";
import { AppBar, Button } from "./components/common";
import { ScanView } from "./components/ScanView";
import { ResultView } from "./components/ResultView";
import { SelectTypePage } from "./components/SelectTypePage";
import { useCamera } from "./hooks/useCamera";
import { useArtifactScanner } from "./hooks/useArtifactScanner";
import type { ViewState, VisitorType } from "./types";

const stateOptions: { id: ViewState; label: string; short: string }[] = [
  { id: "waiting", label: "스캔 대기", short: "A" },
  { id: "recognizing", label: "인식 중", short: "B" },
  { id: "success", label: "결과 + 채팅", short: "C" },
  { id: "not-found", label: "인식 지연", short: "D" },
];

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

  const rescan = () => {
    scanner.reset();
    setViewState("waiting");
  };

  if (!visitorType) {
    return (
      <main className="page">
        <SelectTypePage onSelect={setVisitorType} />
      </main>
    );
  }

  return (
    <main className="page">
      <div className="prototype-header">
        <div>
          <span className="prototype-kicker">MUSEUM AI GUIDE</span>
          <p className="prototype-title">관람의 순간을, 더 깊은 이야기로</p>
        </div>
        <div className="state-switcher" aria-label="화면 상태 선택">
          {stateOptions.map((option) => (
            <Button
              key={option.id}
              className={`state-tab ${viewState === option.id ? "state-tab-active" : ""}`}
              onClick={() => setViewState(option.id)}
            >
              <span>{option.short}</span>
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      <div className={`app-frame ${viewState === "success" ? "success-frame" : ""}`}>
        <AppBar
          modeLabel={modeLabels[visitorType]}
          onBack={() => setVisitorType(null)}
        />
        <div className="frame-content" key={viewState}>
          {viewState === "success" ? (
            <ResultView
              onRescan={rescan}
              photoUrl={scanner.capturedPhoto ?? undefined}
              artifactName={scanner.result?.artifact_name ?? undefined}
              confidence={scanner.result?.confidence ?? undefined}
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
      <p className="prototype-note">
        상태 탭을 선택해 각 화면을 확인하세요. 카메라로 유물을 비추면 자동으로 인식돼요.
      </p>
    </main>
  );
}
