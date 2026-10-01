import { useEffect, useRef, useState } from "react";
import { captureFrame } from "./useCamera";
import { classifyImage, type ClassifyResponse } from "../lib/api";

const POLL_INTERVAL_MS = 1800;
const NOT_FOUND_AFTER_MS = 9000;

export type ScanStatus = "idle" | "scanning" | "found" | "delayed";

export function useArtifactScanner(
  videoElementRef: React.RefObject<HTMLVideoElement | null>,
  active: boolean,
) {
  const [status, setStatus] = useState<ScanStatus>("idle");
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);

  useEffect(() => {
    if (!active) {
      setStatus("idle");
      setConnectionError(false);
      return;
    }

    let cancelled = false;
    let timer: number;
    const startedAt = Date.now();
    setStatus("scanning");
    setResult(null);

    const schedule = () => {
      if (!cancelled) {
        timer = window.setTimeout(tick, POLL_INTERVAL_MS);
      }
    };

    const tick = async () => {
      const video = videoElementRef.current;
      if (!video || video.readyState < 2) {
        schedule();
        return;
      }

      try {
        const blob = await captureFrame(video);
        const response = await classifyImage(blob);
        if (cancelled) return;
        setConnectionError(false);

        if (response.artifact_id) {
          setResult(response);
          setCapturedPhoto(URL.createObjectURL(blob));
          setStatus("found");
          return;
        }

        setStatus(Date.now() - startedAt > NOT_FOUND_AFTER_MS ? "delayed" : "scanning");
      } catch {
        if (!cancelled) setConnectionError(true);
      }
      schedule();
    };

    tick();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active, videoElementRef, retryNonce]);

  const capturedPhotoRef = useRef<string | null>(null);
  useEffect(() => {
    if (capturedPhotoRef.current && capturedPhotoRef.current !== capturedPhoto) {
      URL.revokeObjectURL(capturedPhotoRef.current);
    }
    capturedPhotoRef.current = capturedPhoto;
  }, [capturedPhoto]);

  const restart = () => {
    setResult(null);
    setCapturedPhoto(null);
    setConnectionError(false);
    setRetryNonce((current) => current + 1);
  };

  return {
    status,
    result,
    capturedPhoto,
    connectionError,
    reset: restart,
    retry: restart,
  };
}
