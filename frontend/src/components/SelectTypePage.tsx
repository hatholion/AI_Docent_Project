import { useState, type ReactNode } from "react";
import { Button, Mascot } from "./common";
import mascotImage from "../assets/mascot.png";
import type { VisitorType } from "../types";

import bon002789 from "../assets/gallery/bon002789.jpg";
import duk000798 from "../assets/gallery/duk000798.jpg";
import jub000702 from "../assets/gallery/jub000702.jpg";
import ssu001794 from "../assets/gallery/ssu001794.jpg";
import ssu001846 from "../assets/gallery/ssu001846.jpg";
import ssu003094 from "../assets/gallery/ssu003094.jpg";
import ssu022891 from "../assets/gallery/ssu022891.jpg";

const galleryImages = [
  { src: bon002789, alt: "금동 반가사유상" },
  { src: jub000702, alt: "백자 달항아리" },
  { src: ssu001794, alt: "농경문 청동기" },
  { src: duk000798, alt: "청동촛대" },
  { src: ssu022891, alt: "빗살무늬토기" },
  { src: ssu003094, alt: "요령식 동검" },
  { src: ssu001846, alt: "방패형 동기" },
];

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
  {
    id: "child",
    label: "어린이",
    description: "쉬운 표현과 이야기 중심으로 설명해요.",
    icon: StarIcon,
  },
  {
    id: "general",
    label: "일반",
    description: "핵심 정보와 배경 이야기를 균형 있게 설명해요.",
    icon: PersonIcon,
  },
  {
    id: "expert",
    label: "전문가",
    description: "시대, 재질, 소장품 정보까지 자세히 설명해요.",
    icon: BookIcon,
  },
];

function GalleryPanel() {
  return (
    <div className="select-gallery" aria-hidden="true">
      {galleryImages.map((image) => (
        <div className="select-gallery-tile" key={image.alt}>
          <img src={image.src} alt="" loading="lazy" />
        </div>
      ))}
      <div className="select-gallery-tile select-gallery-mascot">
        <img src={mascotImage} alt="" loading="lazy" />
        <span className="select-gallery-mascot-text">안녕하세요,{"\n"}민속이예요</span>
      </div>
    </div>
  );
}

export function SelectTypePage({
  onSelect,
}: {
  onSelect: (type: VisitorType) => void;
}) {
  const [selected, setSelected] = useState<VisitorType | null>(null);

  return (
    <div className="select-screen">
      <div className="select-view">
        <div className="select-brand">
          <span className="brand-badge">민속톡</span>
          <span className="brand-sub">국립중앙박물관 AI 도슨트</span>
        </div>

        <div className="greeting">
          <Mascot small />
          <p className="greeting-text">
            안녕하세요, 민속이에요. 어떤 방식으로 설명을 들으시겠어요?
          </p>
        </div>

        <h1 className="select-title">관람객 유형을 선택해주세요</h1>
        <p className="select-subtitle">
          관람객에 맞춰 설명의 난이도와 표현 방식을 조절해드립니다.
        </p>

        <div className="visitor-list" role="radiogroup" aria-label="관람객 유형">
          {visitorOptions.map((option) => {
            const Icon = option.icon;
            const isSelected = selected === option.id;
            return (
              <button
                key={option.id}
                type="button"
                role="radio"
                aria-checked={isSelected}
                className={`visitor-row ${isSelected ? "visitor-row-selected" : ""}`}
                onClick={() => setSelected(option.id)}
              >
                <span className="visitor-row-icon">
                  <Icon />
                </span>
                <span className="visitor-row-text">
                  <span className="visitor-row-name">{option.label}</span>
                  <span className="visitor-row-desc">{option.description}</span>
                </span>
                <span className="visitor-row-radio" aria-hidden="true" />
              </button>
            );
          })}
        </div>

        <Button
          className="primary-button start-button"
          onClick={() => selected && onSelect(selected)}
          disabled={!selected}
        >
          시작하기
        </Button>

        <div className="select-footer">
          <span>© 2026 민속톡</span>
          <Button className="help-button" ariaLabel="도움말">
            ?
          </Button>
        </div>
      </div>

      <GalleryPanel />
    </div>
  );
}
