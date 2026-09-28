import { useEffect, useState } from "react";
import { AppBar, Button } from "./components/common";
import { ScanView } from "./components/ScanView";
import { ResultView } from "./components/ResultView";
import type { ViewState } from "./types";

const stateOptions: { id: ViewState; label: string; short: string }[] = [
  { id: "waiting", label: "스캔 대기", short: "A" },
  { id: "recognizing", label: "인식 중", short: "B" },
  { id: "success", label: "결과 + 채팅", short: "C" },
  { id: "not-found", label: "인식 지연", short: "D" },
];

export default function App() {
  const [viewState, setViewState] = useState<ViewState>("waiting");

  useEffect(() => {
    if (viewState !== "recognizing") return;
    const timer = window.setTimeout(() => setViewState("success"), 3600);
    return () => window.clearTimeout(timer);
  }, [viewState]);

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
        <AppBar />
        <div className="frame-content" key={viewState}>
          {viewState === "success" ? (
            <ResultView onRescan={() => setViewState("waiting")} />
          ) : (
            <ScanView
              state={viewState}
              onDismiss={() => setViewState("waiting")}
            />
          )}
        </div>
      </div>
      <p className="prototype-note">
        상태 탭을 선택해 각 화면을 확인하세요. 인식 중 화면은 잠시 후 결과 화면으로
        자동 전환됩니다.
      </p>
    </main>
  );
}
