import type { ReactNode } from "react";
import { Button, Mascot } from "./common";
import type { VisitorType } from "../types";

function StarIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 2.6 5.9 6.4.6-4.8 4.3 1.4 6.2L12 16.9 6.4 20l1.4-6.2L3 9.5l6.4-.6L12 3Z" />
    </svg>
  );
}

function PersonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="8" r="3.4" />
      <path d="M5 20c1.2-4 4-6 7-6s5.8 2 7 6" />
    </svg>
  );
}

function BookIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5C4 4.7 4.7 4 5.5 4H12v16H5.5A1.5 1.5 0 0 1 4 18.5v-13Z" />
      <path d="M20 5.5c0-.8-.7-1.5-1.5-1.5H12v16h6.5a1.5 1.5 0 0 0 1.5-1.5v-13Z" />
    </svg>
  );
}

const visitorOptions: {
  id: VisitorType;
  label: string;
  description: string;
  icon: () => ReactNode;
}[] = [
  { id: "child", label: "어린이", description: "쉽고 재미있게", icon: StarIcon },
  { id: "general", label: "일반인", description: "친절하고 알기 쉽게", icon: PersonIcon },
  {
    id: "expert",
    label: "전문가",
    description: "시대·재질·지정번호까지 자세히",
    icon: BookIcon,
  },
];

export function SelectTypePage({
  onSelect,
}: {
  onSelect: (type: VisitorType) => void;
}) {
  return (
    <div className="select-view">
      <div className="select-brand">
        <span className="brand-badge">민속톡</span>
        <span className="brand-sub">국립중앙박물관 AI 도슨트</span>
      </div>

      <div className="greeting">
        <Mascot />
        <div className="speech-bubble">
          안녕하세요! 저는 민속이에요. 어떤 분이신지 알려주세요.
        </div>
      </div>

      <h1 className="select-title">관람객 유형을 선택해주세요</h1>

      <div className="visitor-grid">
        {visitorOptions.map((option) => {
          const Icon = option.icon;
          return (
            <div key={option.id} className="visitor-card">
              <span className="visitor-icon">
                <Icon />
              </span>
              <p className="visitor-name">{option.label}</p>
              <p className="visitor-desc">{option.description}</p>
              <Button
                className="visitor-select-button"
                onClick={() => onSelect(option.id)}
              >
                선택하기 →
              </Button>
            </div>
          );
        })}
      </div>

      <div className="select-footer">
        <span>© 2026 민속톡</span>
        <Button className="help-button" ariaLabel="도움말">
          ?
        </Button>
      </div>
    </div>
  );
}
