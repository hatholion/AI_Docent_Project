import { useCallback, useEffect, useRef, useState } from "react";

export function useCamera(active: boolean) {
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active) {
      setStream((current) => {
        current?.getTracks().forEach((track) => track.stop());
        return null;
      });
      setError(null);
      return;
    }

    let cancelled = false;
    let activeStream: MediaStream | null = null;

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((mediaStream) => {
        if (cancelled) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }
        activeStream = mediaStream;
        setStream(mediaStream);
      })
      .catch(() => {
        if (!cancelled) {
          setError("카메라를 사용할 수 없어요. 브라우저 카메라 권한을 확인해주세요.");
        }
      });

    return () => {
      cancelled = true;
      activeStream?.getTracks().forEach((track) => track.stop());
    };
  }, [active]);

  const attachVideo = useCallback(
    (video: HTMLVideoElement | null) => {
      videoElementRef.current = video;
      if (video && stream) {
        video.srcObject = stream;
      }
    },
    [stream],
  );

  return { videoElementRef, attachVideo, error };
}

export function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    return Promise.reject(new Error("캔버스를 사용할 수 없어요."));
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("프레임 캡처에 실패했어요."))),
      "image/jpeg",
      0.85,
    );
  });
}
