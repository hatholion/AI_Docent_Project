import { ReactNode } from "react";
import mascotImage from "../assets/mascot.png";

export const artifactPhoto =
  "https://images.unsplash.com/photo-1756308480720-19ef2f35fa0d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwzfHxLb3JlYW4lMjBtdXNldW0lMjBidWRkaGlzdCUyMHN0YXR1ZSUyMHNjdWxwdHVyZSUyMGV4aGliaXRpb258ZW58MXx8fHwxNzkwNTYxNDM1fDA&ixlib=rb-4.1.0&q=80&w=1080";

export function ArrowLeftIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m15 18-6-6 6-6" />
    </svg>
  );
}

export function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m4 4 17 8-17 8 3-8-3-8Z" />
      <path d="M7 12h14" />
    </svg>
  );
}

export function XIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m6 6 12 12M18 6 6 18" />
    </svg>
  );
}

export function ScanIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 9V6a2 2 0 0 1 2-2h3M15 4h3a2 2 0 0 1 2 2v3M20 15v3a2 2 0 0 1-2 2h-3M9 20H6a2 2 0 0 1-2-2v-3" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function Button({
  children,
  className = "",
  type = "button",
  onClick,
  ariaLabel,
}: {
  children: ReactNode;
  className?: string;
  type?: "button" | "submit";
  onClick?: () => void;
  ariaLabel?: string;
}) {
  return (
    <button
      type={type}
      className={className}
      onClick={onClick}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}

export function Mascot({ small = false }: { small?: boolean }) {
  return (
    <div className={`mascot ${small ? "mascot-small" : ""}`}>
      <img className="mascot-image" src={mascotImage} alt="민속이" />
    </div>
  );
}

export function AppBar({
  modeLabel,
  onBack,
}: {
  modeLabel: string;
  onBack?: () => void;
}) {
  return (
    <div className="app-bar">
      <Button className="icon-button" onClick={onBack} ariaLabel="뒤로가기">
        <ArrowLeftIcon />
      </Button>
      <div className="brand">
        <Mascot small />
        <span>민속톡</span>
      </div>
      <span className="mode-badge">{modeLabel}</span>
    </div>
  );
}
