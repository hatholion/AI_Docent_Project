export default function MuseumBackgroundRealistic() {
  return (
    <div
      className="fixed inset-0 pointer-events-none select-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    >
    <svg
      viewBox="0 0 1440 1024"
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        {/* ── Texture: 미세 노이즈 (한지·나무 질감) ── */}
        <filter id="grain" x="0%" y="0%" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="4" stitchTiles="stitch" result="noise" />
          <feColorMatrix type="saturate" values="0" in="noise" result="grey" />
          <feBlend in="SourceGraphic" in2="grey" mode="overlay" result="blend" />
          <feComposite in="blend" in2="SourceGraphic" operator="in" />
        </filter>

        {/* ── 나무 결 텍스처 ── */}
        <filter id="wood" x="0%" y="0%" width="100%" height="100%">
          <feTurbulence type="turbulence" baseFrequency="0.015 0.25" numOctaves="4" seed="3" result="t" />
          <feColorMatrix type="matrix" values="0 0 0 0 0.55  0 0 0 0 0.38  0 0 0 0 0.22  0 0 0 0.18 0" in="t" result="wc" />
          <feBlend in="SourceGraphic" in2="wc" mode="multiply" />
        </filter>

        {/* ── 피사계심도 블러 (옆면 요소 흐림) ── */}
        <filter id="blur-far" x="-5%" y="-5%" width="110%" height="110%">
          <feGaussianBlur stdDeviation="2.5" />
        </filter>
        <filter id="blur-near" x="-5%" y="-5%" width="110%" height="110%">
          <feGaussianBlur stdDeviation="1.2" />
        </filter>

        {/* ── 스포트라이트 글로우 ── */}
        <filter id="glow-warm" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="18" result="b" />
          <feColorMatrix type="matrix" values="1 0.2 0 0 0  0 0.7 0 0 0  0 0 0 0 0  0 0 0 0.7 0" in="b" result="c" />
          <feBlend in="SourceGraphic" in2="c" mode="screen" />
        </filter>
        <filter id="glow-soft" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="12" result="b" />
          <feBlend in="SourceGraphic" in2="b" mode="screen" />
        </filter>

        {/* ── 그라데이션 정의 ── */}
        {/* 전체 배경 그라데이션 */}
        <radialGradient id="bg-radial" cx="50%" cy="42%" r="65%">
          <stop offset="0%" stopColor="#F8EDD8" />
          <stop offset="55%" stopColor="#EDD9B8" />
          <stop offset="100%" stopColor="#C8A47A" />
        </radialGradient>

        {/* 천장 그라데이션 */}
        <linearGradient id="ceiling-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2A1A0A" />
          <stop offset="100%" stopColor="#5C3A1E" />
        </linearGradient>

        {/* 바닥 그라데이션 */}
        <linearGradient id="floor-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#A0722A" stopOpacity="0.9" />
          <stop offset="40%" stopColor="#8C6239" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#5C3A18" stopOpacity="0.95" />
        </linearGradient>

        {/* 중앙 스포트라이트 (밝은 원형 광원) */}
        <radialGradient id="center-light" cx="50%" cy="38%" r="32%">
          <stop offset="0%" stopColor="#FFF5E0" stopOpacity="0.92" />
          <stop offset="40%" stopColor="#F8E8C8" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#F3E9DC" stopOpacity="0" />
        </radialGradient>

        {/* 좌측 벽 그라데이션 */}
        <linearGradient id="left-wall-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#3D2210" />
          <stop offset="70%" stopColor="#7A5030" />
          <stop offset="100%" stopColor="#C9A67E" stopOpacity="0.3" />
        </linearGradient>

        {/* 우측 벽 그라데이션 */}
        <linearGradient id="right-wall-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#C9A67E" stopOpacity="0.3" />
          <stop offset="30%" stopColor="#7A5030" />
          <stop offset="100%" stopColor="#3D2210" />
        </linearGradient>

        {/* 스포트라이트 콘 */}
        <radialGradient id="spot-l" cx="50%" cy="0%" r="100%" fx="50%" fy="0%">
          <stop offset="0%" stopColor="#FFF3D0" stopOpacity="0.85" />
          <stop offset="60%" stopColor="#F5DBA0" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#E8C87A" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="spot-r" cx="50%" cy="0%" r="100%" fx="50%" fy="0%">
          <stop offset="0%" stopColor="#FFF3D0" stopOpacity="0.85" />
          <stop offset="60%" stopColor="#F5DBA0" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#E8C87A" stopOpacity="0" />
        </radialGradient>

        {/* 유물 글로우 */}
        <radialGradient id="artifact-glow-l" cx="50%" cy="80%" r="60%">
          <stop offset="0%" stopColor="#FFF8E8" stopOpacity="0.9" />
          <stop offset="50%" stopColor="#F5E4C0" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#C9A67E" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="artifact-glow-r" cx="50%" cy="80%" r="60%">
          <stop offset="0%" stopColor="#FFF8E8" stopOpacity="0.9" />
          <stop offset="50%" stopColor="#F5E4C0" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#C9A67E" stopOpacity="0" />
        </radialGradient>

        {/* 창호 패턴 */}
        <pattern id="changho" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
          <rect width="24" height="24" fill="none" />
          <rect x="0" y="0" width="24" height="1.5" fill="#C9A67E" opacity="0.6" />
          <rect x="0" y="0" width="1.5" height="24" fill="#C9A67E" opacity="0.6" />
          <rect x="12" y="0" width="1" height="24" fill="#C9A67E" opacity="0.35" />
          <rect x="0" y="12" width="24" height="1" fill="#C9A67E" opacity="0.35" />
        </pattern>

        {/* 복도 바닥 타일 */}
        <pattern id="floor-tile" x="0" y="0" width="80" height="80" patternUnits="userSpaceOnUse">
          <rect width="80" height="80" fill="none" />
          <rect x="1" y="1" width="78" height="78" fill="none" stroke="#8C6239" strokeWidth="0.8" opacity="0.25" />
          <rect x="10" y="10" width="60" height="60" fill="none" stroke="#C9A67E" strokeWidth="0.4" opacity="0.15" />
        </pattern>

        {/* 클리핑 마스크 (원근 변환용) */}
        <clipPath id="scene-clip">
          <rect width="1440" height="1024" />
        </clipPath>

        {/* 중앙 부분 마스킹 (UI 영역) */}
        <mask id="center-mask">
          <rect width="1440" height="1024" fill="white" />
          <rect x="340" y="0" width="760" height="1024" fill="black" opacity="0.18" />
        </mask>
      </defs>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 0 · 기본 배경 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <rect width="1440" height="1024" fill="url(#bg-radial)" />

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 1 · 원근감 있는 전시관 내부 구조 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}

      {/* 바닥 면 (원근) */}
      <polygon
        points="0,1024 1440,1024 1100,580 340,580"
        fill="url(#floor-grad)"
        filter="url(#wood)"
        opacity="0.95"
      />
      {/* 바닥 타일 패턴 오버레이 */}
      <polygon
        points="0,1024 1440,1024 1100,580 340,580"
        fill="url(#floor-tile)"
        opacity="0.4"
      />
      {/* 바닥 반사 하이라이트 */}
      <polygon
        points="480,1024 960,1024 860,700 580,700"
        fill="url(#center-light)"
        opacity="0.35"
      />

      {/* 좌측 벽 (원근) */}
      <polygon
        points="0,0 340,580 340,1024 0,1024"
        fill="url(#left-wall-grad)"
        opacity="0.92"
      />
      {/* 우측 벽 (원근) */}
      <polygon
        points="1440,0 1100,580 1100,1024 1440,1024"
        fill="url(#right-wall-grad)"
        opacity="0.92"
      />

      {/* 천장 면 (원근) */}
      <polygon
        points="0,0 1440,0 1100,580 340,580"
        fill="url(#ceiling-grad)"
        opacity="0.88"
      />

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 2 · 한옥 기둥 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}

      {/* 왼쪽 앞 기둥 */}
      <g filter="url(#blur-near)">
        <rect x="82" y="320" width="52" height="704" rx="4" fill="#2E1A08" opacity="0.95" />
        {/* 기둥 하이라이트 */}
        <rect x="90" y="320" width="12" height="704" rx="2" fill="#8C6239" opacity="0.4" />
        {/* 기둥 기단부 */}
        <ellipse cx="108" cy="1010" rx="32" ry="8" fill="#1A0E04" opacity="0.7" />
        {/* 기둥 주두 */}
        <rect x="68" y="316" width="80" height="20" rx="3" fill="#3D2210" opacity="0.9" />
        <rect x="60" y="300" width="96" height="18" rx="3" fill="#4A2A12" opacity="0.85" />
      </g>

      {/* 왼쪽 뒤 기둥 */}
      <g filter="url(#blur-far)">
        <rect x="228" y="420" width="36" height="604" rx="3" fill="#3D2210" opacity="0.85" />
        <rect x="234" y="420" width="8" height="604" rx="2" fill="#7A5030" opacity="0.35" />
        <rect x="216" y="416" width="60" height="14" rx="2" fill="#3D2210" opacity="0.8" />
      </g>

      {/* 오른쪽 앞 기둥 */}
      <g filter="url(#blur-near)">
        <rect x="1306" y="320" width="52" height="704" rx="4" fill="#2E1A08" opacity="0.95" />
        <rect x="1312" y="320" width="12" height="704" rx="2" fill="#8C6239" opacity="0.4" />
        <ellipse cx="1332" cy="1010" rx="32" ry="8" fill="#1A0E04" opacity="0.7" />
        <rect x="1292" y="316" width="80" height="20" rx="3" fill="#3D2210" opacity="0.9" />
        <rect x="1284" y="300" width="96" height="18" rx="3" fill="#4A2A12" opacity="0.85" />
      </g>

      {/* 오른쪽 뒤 기둥 */}
      <g filter="url(#blur-far)">
        <rect x="1176" y="420" width="36" height="604" rx="3" fill="#3D2210" opacity="0.85" />
        <rect x="1182" y="420" width="8" height="604" rx="2" fill="#7A5030" opacity="0.35" />
        <rect x="1164" y="416" width="60" height="14" rx="2" fill="#3D2210" opacity="0.8" />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 3 · 격자 창호 (좌측 벽) */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <g filter="url(#blur-far)">
        {/* 창호 프레임 */}
        <rect x="-20" y="250" width="160" height="420" rx="4" fill="#1A0E04" opacity="0.7" />
        <rect x="-14" y="256" width="148" height="408" rx="2" fill="#3D1F08" opacity="0.6" />
        {/* 창호 격자 패턴 */}
        <rect x="-14" y="256" width="148" height="408" fill="url(#changho)" opacity="0.55" />
        {/* 창호 빛 투과 */}
        <rect x="-14" y="256" width="148" height="408" rx="2" fill="url(#spot-l)" opacity="0.5" />
        {/* 창호 외곽 테두리 */}
        <rect x="-14" y="256" width="148" height="408" rx="2" fill="none" stroke="#C9A67E" strokeWidth="2" opacity="0.4" />

        {/* 창호 중간 가로 구분대 */}
        <rect x="-14" y="460" width="148" height="4" fill="#5C3A1E" opacity="0.7" />

        {/* 창호에서 새어 나오는 빛 */}
        <polygon
          points="-14,256 134,256 230,100 -80,100"
          fill="#FFF8E8"
          opacity="0.08"
        />
      </g>

      {/* 우측 창호 */}
      <g filter="url(#blur-far)">
        <rect x="1300" y="250" width="160" height="420" rx="4" fill="#1A0E04" opacity="0.7" />
        <rect x="1306" y="256" width="148" height="408" rx="2" fill="#3D1F08" opacity="0.6" />
        <rect x="1306" y="256" width="148" height="408" fill="url(#changho)" opacity="0.55" />
        <rect x="1306" y="256" width="148" height="408" rx="2" fill="url(#spot-r)" opacity="0.5" />
        <rect x="1306" y="256" width="148" height="408" rx="2" fill="none" stroke="#C9A67E" strokeWidth="2" opacity="0.4" />
        <rect x="1306" y="460" width="148" height="4" fill="#5C3A1E" opacity="0.7" />
        <polygon
          points="1306,256 1454,256 1520,100 1210,100"
          fill="#FFF8E8"
          opacity="0.08"
        />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 4 · 천장 보 / 처마 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}

      {/* 메인 처마 보 (상단 가로) */}
      <g>
        <rect x="0" y="0" width="1440" height="48" fill="#1A0E04" opacity="0.95" />
        {/* 처마 하단면 */}
        <polygon
          points="0,48 1440,48 1100,580 340,580"
          fill="#2A1608"
          opacity="0.6"
        />
        {/* 보 장식선 */}
        <rect x="0" y="44" width="1440" height="3" fill="#8C6239" opacity="0.35" />
        <rect x="0" y="48" width="1440" height="2" fill="#C9A67E" opacity="0.2" />
      </g>

      {/* 좌측 사선 보 */}
      <polygon
        points="0,0 340,580 300,580 0,48"
        fill="#22120A"
        opacity="0.88"
      />
      <polygon
        points="0,48 300,580 280,580 8,48"
        fill="#8C6239"
        opacity="0.12"
      />

      {/* 우측 사선 보 */}
      <polygon
        points="1440,0 1100,580 1140,580 1440,48"
        fill="#22120A"
        opacity="0.88"
      />
      <polygon
        points="1440,48 1140,580 1160,580 1432,48"
        fill="#8C6239"
        opacity="0.12"
      />

      {/* 천장 가로 보 (안쪽) */}
      <g filter="url(#blur-far)" opacity="0.6">
        {[0.2, 0.4, 0.6, 0.8].map((t, i) => {
          const x0 = 340 * t;
          const x1 = 1440 - 340 * t;
          const y = 48 + (580 - 48) * t * 0.5;
          return (
            <line
              key={i}
              x1={x0} y1={y}
              x2={x1} y2={y}
              stroke="#5C3A1E"
              strokeWidth={3 - i * 0.5}
              opacity={0.5 - i * 0.08}
            />
          );
        })}
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 5 · 스포트라이트 콘 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}

      {/* 좌측 스포트라이트 */}
      <g opacity="0.65" filter="url(#glow-soft)">
        <polygon
          points="160,48 180,48 240,820 120,820"
          fill="url(#spot-l)"
          opacity="0.55"
        />
        {/* 광원 포인트 */}
        <ellipse cx="170" cy="52" rx="10" ry="6" fill="#FFFAE0" opacity="0.9" />
      </g>

      {/* 우측 스포트라이트 */}
      <g opacity="0.65" filter="url(#glow-soft)">
        <polygon
          points="1260,48 1280,48 1340,820 1220,820"
          fill="url(#spot-r)"
          opacity="0.55"
        />
        <ellipse cx="1270" cy="52" rx="10" ry="6" fill="#FFFAE0" opacity="0.9" />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 6 · 좌측 유물 + 좌대 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <g filter="url(#blur-near)">
        {/* 유물 글로우 (스포트라이트 웅덩이) */}
        <ellipse cx="168" cy="750" rx="120" ry="40" fill="url(#artifact-glow-l)" opacity="0.7" />

        {/* 좌대 (전시 받침대) */}
        {/* 상판 */}
        <rect x="100" y="710" width="138" height="14" rx="3" fill="#5C3A1E" />
        <rect x="96" y="708" width="146" height="8" rx="2" fill="#7A5030" opacity="0.9" />
        {/* 몸통 */}
        <rect x="118" y="724" width="102" height="200" fill="#3D2210" />
        {/* 몸통 하이라이트 */}
        <rect x="118" y="724" width="16" height="200" fill="#7A5030" opacity="0.3" />
        {/* 하단 받침 */}
        <rect x="108" y="916" width="122" height="16" rx="2" fill="#4A2A12" />
        <rect x="102" y="928" width="134" height="10" rx="2" fill="#3D2210" />

        {/* 바닥 그림자 */}
        <ellipse cx="169" cy="940" rx="72" ry="10" fill="#1A0E04" opacity="0.45" />

        {/* ── 백자 달항아리 실루엣 ── */}
        {/* 항아리 몸체 */}
        <ellipse cx="169" cy="640" rx="52" ry="62" fill="#F0EDE6" opacity="0.88" />
        {/* 항아리 목 */}
        <ellipse cx="169" cy="580" rx="20" ry="10" fill="#E8E4DC" opacity="0.88" />
        <rect x="149" y="580" width="40" height="38" fill="#E8E4DC" opacity="0.88" />
        {/* 항아리 구연부 */}
        <ellipse cx="169" cy="578" rx="22" ry="8" fill="#DDD9D0" opacity="0.85" />
        {/* 항아리 하단부 */}
        <ellipse cx="169" cy="700" rx="42" ry="13" fill="#DDD9D0" opacity="0.82" />
        {/* 항아리 광택 하이라이트 */}
        <ellipse cx="155" cy="610" rx="14" ry="22" fill="white" opacity="0.35" transform="rotate(-20 155 610)" />
        <ellipse cx="152" cy="605" rx="5" ry="9" fill="white" opacity="0.5" transform="rotate(-20 152 605)" />
        {/* 항아리 그림자 */}
        <ellipse cx="169" cy="702" rx="52" ry="8" fill="#3D2210" opacity="0.3" />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 7 · 우측 유물 + 좌대 (청자) */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <g filter="url(#blur-near)">
        {/* 글로우 */}
        <ellipse cx="1272" cy="750" rx="120" ry="40" fill="url(#artifact-glow-r)" opacity="0.7" />

        {/* 좌대 */}
        <rect x="1204" y="710" width="138" height="14" rx="3" fill="#5C3A1E" />
        <rect x="1200" y="708" width="146" height="8" rx="2" fill="#7A5030" opacity="0.9" />
        <rect x="1222" y="724" width="102" height="200" fill="#3D2210" />
        <rect x="1222" y="724" width="16" height="200" fill="#7A5030" opacity="0.3" />
        <rect x="1212" y="916" width="122" height="16" rx="2" fill="#4A2A12" />
        <rect x="1206" y="928" width="134" height="10" rx="2" fill="#3D2210" />
        <ellipse cx="1273" cy="940" rx="72" ry="10" fill="#1A0E04" opacity="0.45" />

        {/* ── 청자 매병 실루엣 ── */}
        {/* 병 몸체 */}
        <path
          d="M1230,688 C1230,688 1215,660 1215,630 C1215,590 1225,570 1235,560 L1245,555 C1245,555 1250,548 1273,548 C1296,548 1301,555 1301,555 L1311,560 C1321,570 1331,590 1331,630 C1331,660 1316,688 1316,688 Z"
          fill="#8BA888"
          opacity="0.85"
        />
        {/* 병 목 */}
        <rect x="1258" y="530" width="30" height="28" rx="4" fill="#7A9878" opacity="0.85" />
        {/* 구연부 */}
        <ellipse cx="1273" cy="530" rx="18" ry="7" fill="#6A8868" opacity="0.85" />
        {/* 광택 하이라이트 */}
        <ellipse cx="1250" cy="600" rx="10" ry="30" fill="white" opacity="0.22" transform="rotate(-12 1250 600)" />
        <ellipse cx="1248" cy="592" rx="4" ry="12" fill="white" opacity="0.38" transform="rotate(-12 1248 592)" />
        {/* 청자 무늬 (은은한 음각선) */}
        <ellipse cx="1273" cy="628" rx="42" ry="52" fill="none" stroke="#5A7858" strokeWidth="1" opacity="0.4" />
        <path d="M1240,590 Q1273,570 1306,590" fill="none" stroke="#5A7858" strokeWidth="0.8" opacity="0.3" />
        {/* 그림자 */}
        <ellipse cx="1273" cy="690" rx="48" ry="8" fill="#3D2210" opacity="0.3" />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 8 · 안쪽 전시 공간 (원경) */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <g filter="url(#blur-far)" opacity="0.55">
        {/* 안쪽 벽 */}
        <rect x="340" y="240" width="760" height="340" fill="#C8A47A" opacity="0.3" rx="2" />

        {/* 안쪽 창호 좌 */}
        <rect x="370" y="280" width="100" height="200" rx="3" fill="#3D2210" opacity="0.55" />
        <rect x="376" y="286" width="88" height="188" fill="url(#changho)" opacity="0.7" />
        <rect x="376" y="286" width="88" height="188" fill="#FFF8E8" opacity="0.12" />

        {/* 안쪽 창호 우 */}
        <rect x="970" y="280" width="100" height="200" rx="3" fill="#3D2210" opacity="0.55" />
        <rect x="976" y="286" width="88" height="188" fill="url(#changho)" opacity="0.7" />
        <rect x="976" y="286" width="88" height="188" fill="#FFF8E8" opacity="0.12" />

        {/* 안쪽 반가사유상 실루엣 */}
        <g opacity="0.6">
          {/* 좌대 */}
          <rect x="680" y="460" width="80" height="80" rx="4" fill="#5C3A1E" opacity="0.7" />
          <rect x="672" y="456" width="96" height="10" rx="2" fill="#7A5030" opacity="0.7" />
          {/* 불상 몸체 */}
          <ellipse cx="720" cy="390" rx="30" ry="40" fill="#8C7050" opacity="0.55" />
          {/* 불상 머리 */}
          <ellipse cx="720" cy="345" rx="18" ry="20" fill="#8C7050" opacity="0.55" />
          {/* 반가사유상 특징: 다리 올린 자세 */}
          <path d="M700,430 Q690,450 700,460 Q720,465 740,460 Q750,450 740,430" fill="#7A6040" opacity="0.5" />
          {/* 팔 (사유 자세) */}
          <path d="M710,390 Q700,370 705,358 Q710,352 716,355" fill="none" stroke="#7A6040" strokeWidth="8" strokeLinecap="round" opacity="0.5" />
          {/* 글로우 */}
          <ellipse cx="720" cy="410" rx="50" ry="70" fill="#FFF5E0" opacity="0.12" />
        </g>

        {/* 안쪽 천장 조명 */}
        <ellipse cx="720" cy="320" rx="60" ry="20" fill="#FFF8E0" opacity="0.2" />
      </g>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LAYER 9 · 분위기 오버레이 */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}

      {/* 전체 웜톤 오버레이 */}
      <rect width="1440" height="1024" fill="#8C6239" opacity="0.06" />

      {/* 중앙 밝은 영역 (UI 카드가 올라갈 자리) */}
      <rect x="340" y="0" width="760" height="1024" fill="url(#center-light)" opacity="0.28" />

      {/* 좌우 가장자리 비네팅 */}
      <rect x="0" y="0" width="200" height="1024"
        fill="url(#left-wall-grad)" opacity="0.25" />
      <rect x="1240" y="0" width="200" height="1024"
        fill="url(#right-wall-grad)" opacity="0.25" />

      {/* 상단 비네팅 */}
      <linearGradient id="vign-top" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#1A0E04" stopOpacity="0.7" />
        <stop offset="100%" stopColor="#1A0E04" stopOpacity="0" />
      </linearGradient>
      <rect x="0" y="0" width="1440" height="180" fill="url(#vign-top)" />

      {/* 하단 비네팅 */}
      <linearGradient id="vign-bot" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#5C3A1E" stopOpacity="0" />
        <stop offset="100%" stopColor="#3D2210" stopOpacity="0.6" />
      </linearGradient>
      <rect x="0" y="820" width="1440" height="204" fill="url(#vign-bot)" />

      {/* 미세 그레인 노이즈 (전체 필름감) */}
      <rect width="1440" height="1024" fill="url(#bg-radial)" opacity="0" filter="url(#grain)" />
      <rect width="1440" height="1024" fill="#F3E9DC" opacity="0.04" filter="url(#grain)" />
    </svg>
    </div>
  );
}
