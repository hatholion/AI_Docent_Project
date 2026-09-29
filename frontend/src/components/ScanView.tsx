import type { RefCallback } from "react";
import { Button, ErrorToast, Mascot, ScanIcon, XIcon } from "./common";
import type { ViewState } from "../types";

function GuideFrame({ active = false }: { active?: boolean }) {
  return (
    <div className={`guide-frame ${active ? "guide-frame-active" : ""}`}>
      <span className="corner corner-tl" />
      <span className="corner corner-tr" />
      <span className="corner corner-bl" />
      <span className="corner corner-br" />
      {!active && (
        <div className="auto-badge">
          <ScanIcon />
          자동 인식 모드
        </div>
      )}
    </div>
  );
}

function CameraSurface({
  state,
  onDismiss,
  attachVideo,
  cameraError,
}: {
  state: Exclude<ViewState, "success">;
  onDismiss: () => void;
  attachVideo: RefCallback<HTMLVideoElement>;
  cameraError: string | null;
}) {
  const recognizing = state === "recognizing";
  const delayed = state === "not-found";

  return (
    <div className="camera-shell">
      {cameraError ? (
        <div className="camera-error">
          <p>{cameraError}</p>
        </div>
      ) : (
        <video
          ref={attachVideo}
          className="camera-feed"
          autoPlay
          playsInline
          muted
        />
      )}
      <div className="camera-shade" />
      <GuideFrame active={recognizing} />

      {recognizing && (
        <div className="recognition-status">
          <div className="floating-mascot">
            <Mascot small />
          </div>
          <div>
            <span className="eyebrow">AI 분석 중</span>
            <p>민속이가 유물을 살펴보고 있어요</p>
          </div>
          <span className="loading-dots" aria-label="로딩 중">
            <i />
            <i />
            <i />
          </span>
        </div>
      )}

      {delayed && (
        <div className="delay-card">
          <Button className="close-button" onClick={onDismiss} ariaLabel="안내 닫기">
            <XIcon />
          </Button>
          <div className="delay-title">
            <Mascot small />
            <div>
              <span className="eyebrow">스캔 도움말</span>
              <p>아직 유물을 인식하지 못했어요</p>
            </div>
          </div>
          <ul>
            <li>조금 더 가까이 비춰주세요</li>
            <li>밝은 곳에서 비춰주세요</li>
          </ul>
          <span className="scan-still-running">자동 스캔을 계속하고 있어요</span>
        </div>
      )}
    </div>
  );
}

export function ScanView({
  state,
  onDismiss,
  attachVideo,
  cameraError,
  connectionError,
  onRetry,
}: {
  state: "waiting" | "recognizing" | "not-found";
  onDismiss: () => void;
  attachVideo: RefCallback<HTMLVideoElement>;
  cameraError: string | null;
  connectionError: boolean;
  onRetry: () => void;
}) {
  const recognizing = state === "recognizing";
  const delayed = state === "not-found";

  return (
    <div className="scan-view">
      <div className="scan-copy">
        <span className="section-kicker">
          {recognizing ? "유물 인식 중" : delayed ? "자동 스캔 중" : "카메라 스캔"}
        </span>
        <p className="scan-title">
          {recognizing
            ? "잠시만 기다려주세요"
            : delayed
              ? "각도를 천천히 바꿔보세요"
              : "유물을 프레임 안에 비춰주세요"}
        </p>
        <p className="scan-subtitle">
          {recognizing
            ? "형태와 특징을 안전하게 분석하고 있어요"
            : "버튼을 누르지 않아도 자동으로 인식돼요"}
        </p>
      </div>

      <CameraSurface
        state={state}
        onDismiss={onDismiss}
        attachVideo={attachVideo}
        cameraError={cameraError}
      />

      {!recognizing && !delayed && (
        <div className="tip-card">
          <Mascot />
          <div>
            <span>민속이의 관람 팁</span>
            <p>유물 정면이 잘 보이는 밝은 곳에서 비춰주세요</p>
          </div>
        </div>
      )}

      {connectionError && (
        <ErrorToast
          message="서버에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
          onRetry={onRetry}
        />
      )}
    </div>
  );
}
