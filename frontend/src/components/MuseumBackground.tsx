const MAIN = '#8C6239';
const LIGHT = '#C9A67E';
const PALE = '#F3E9DC';
const BG = '#FBF6EF';

// ── 달항아리 ──
function MoonJar({ cx, cy, size, op }: { cx: number; cy: number; size: number; op: number }) {
  return (
    <g opacity={op} fill={MAIN}>
      <ellipse cx={cx} cy={cy} rx={size * 0.48} ry={size * 0.46} />
      <ellipse cx={cx} cy={cy - size * 0.42} rx={size * 0.16} ry={size * 0.07} />
      <ellipse cx={cx} cy={cy + size * 0.44} rx={size * 0.22} ry={size * 0.06} />
    </g>
  );
}

// ── 청자 병 ──
function TallVase({ cx, cy, size, op }: { cx: number; cy: number; size: number; op: number }) {
  const s = size;
  return (
    <g opacity={op} fill={MAIN}>
      <path
        d={`M ${cx},${cy - s * 0.5} C ${cx - s * 0.06},${cy - s * 0.46} ${cx - s * 0.07},${cy - s * 0.36} ${cx - s * 0.07},${cy - s * 0.28} C ${cx - s * 0.07},${cy - s * 0.18} ${cx - s * 0.22},${cy - s * 0.04} ${cx - s * 0.22},${cy + s * 0.12} C ${cx - s * 0.22},${cy + s * 0.36} ${cx - s * 0.14},${cy + s * 0.5} ${cx},${cy + s * 0.5} C ${cx + s * 0.14},${cy + s * 0.5} ${cx + s * 0.22},${cy + s * 0.36} ${cx + s * 0.22},${cy + s * 0.12} C ${cx + s * 0.22},${cy - s * 0.04} ${cx + s * 0.07},${cy - s * 0.18} ${cx + s * 0.07},${cy - s * 0.28} C ${cx + s * 0.07},${cy - s * 0.36} ${cx + s * 0.06},${cy - s * 0.46} ${cx},${cy - s * 0.5} Z`}
      />
      <ellipse cx={cx} cy={cy + s * 0.5} rx={s * 0.13} ry={s * 0.04} />
      <ellipse cx={cx} cy={cy - s * 0.5} rx={s * 0.065} ry={s * 0.025} />
    </g>
  );
}

// ── 불상 ──
function Buddha({ cx, cy, size, op }: { cx: number; cy: number; size: number; op: number }) {
  const s = size;
  return (
    <g opacity={op} fill={MAIN}>
      {/* 광배 (halo) */}
      <circle
        cx={cx}
        cy={cy - s * 0.52}
        r={s * 0.3}
        fill="none"
        stroke={MAIN}
        strokeWidth={s * 0.03}
      />
      {/* 머리 */}
      <circle cx={cx} cy={cy - s * 0.52} r={s * 0.17} />
      {/* 육계 (topknot) */}
      <ellipse cx={cx} cy={cy - s * 0.7} rx={s * 0.07} ry={s * 0.055} />
      {/* 몸 */}
      <ellipse cx={cx} cy={cy} rx={s * 0.28} ry={s * 0.42} />
      {/* 가부좌 */}
      <ellipse cx={cx} cy={cy + s * 0.38} rx={s * 0.38} ry={s * 0.1} />
    </g>
  );
}

// ── 향로 ──
function Incenser({ cx, cy, size, op }: { cx: number; cy: number; size: number; op: number }) {
  const s = size;
  return (
    <g opacity={op} fill={MAIN}>
      <path
        d={`M ${cx - s * 0.32},${cy + s * 0.04} C ${cx - s * 0.36},${cy - s * 0.26} ${cx - s * 0.2},${cy - s * 0.38} ${cx},${cy - s * 0.38} C ${cx + s * 0.2},${cy - s * 0.38} ${cx + s * 0.36},${cy - s * 0.26} ${cx + s * 0.32},${cy + s * 0.04} Z`}
      />
      <ellipse cx={cx} cy={cy + s * 0.04} rx={s * 0.33} ry={s * 0.08} />
      <rect
        x={cx - s * 0.055}
        y={cy + s * 0.1}
        width={s * 0.11}
        height={s * 0.24}
        rx={s * 0.025}
      />
      <ellipse cx={cx} cy={cy + s * 0.34} rx={s * 0.21} ry={s * 0.06} />
    </g>
  );
}

// ── 전시 좌대 ──
function Pedestal({
  x,
  y,
  w,
  h,
  op,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  op: number;
}) {
  const ext = w * 0.12;
  return (
    <g opacity={op}>
      <rect x={x - ext} y={y - 18} width={w + ext * 2} height={18} rx={3} fill={MAIN} />
      <rect x={x} y={y} width={w} height={h} fill={LIGHT} />
      <rect x={x} y={y} width={w * 0.07} height={h} fill={MAIN} opacity={0.25} />
      <rect x={x + w * 0.93} y={y} width={w * 0.07} height={h} fill={MAIN} opacity={0.25} />
      <rect x={x - 7} y={y + h} width={w + 14} height={16} rx={3} fill={MAIN} />
    </g>
  );
}

// ── 민속이 마스코트 ──
function Minsoki({ cx, cy, s, op }: { cx: number; cy: number; s: number; op: number }) {
  return (
    <g opacity={op}>
      {/* 그림자 */}
      <ellipse cx={cx} cy={cy + s * 0.74} rx={s * 0.3} ry={s * 0.065} fill={MAIN} opacity={0.18} />
      {/* 두루마기 몸 */}
      <ellipse cx={cx} cy={cy + s * 0.38} rx={s * 0.27} ry={s * 0.35} fill={MAIN} />
      {/* 깃 */}
      <path
        d={`M ${cx - s * 0.12},${cy + s * 0.2} L ${cx},${cy + s * 0.3} L ${cx + s * 0.12},${cy + s * 0.2}`}
        fill="none"
        stroke={PALE}
        strokeWidth={s * 0.025}
      />
      {/* 머리 */}
      <circle cx={cx} cy={cy} r={s * 0.22} fill={LIGHT} />
      {/* 눈 */}
      <circle cx={cx - s * 0.08} cy={cy - s * 0.01} r={s * 0.033} fill={MAIN} />
      <circle cx={cx + s * 0.08} cy={cy - s * 0.01} r={s * 0.033} fill={MAIN} />
      {/* 웃음 */}
      <path
        d={`M ${cx - s * 0.07},${cy + s * 0.07} Q ${cx},${cy + s * 0.13} ${cx + s * 0.07},${cy + s * 0.07}`}
        fill="none"
        stroke={MAIN}
        strokeWidth={s * 0.022}
        strokeLinecap="round"
      />
      {/* 볼터치 */}
      <ellipse cx={cx - s * 0.14} cy={cy + s * 0.04} rx={s * 0.055} ry={s * 0.033} fill={MAIN} opacity={0.22} />
      <ellipse cx={cx + s * 0.14} cy={cy + s * 0.04} rx={s * 0.055} ry={s * 0.033} fill={MAIN} opacity={0.22} />
      {/* 팔 */}
      <path
        d={`M ${cx - s * 0.26},${cy + s * 0.22} Q ${cx - s * 0.45},${cy + s * 0.15} ${cx - s * 0.41},${cy + s * 0.02}`}
        fill="none"
        stroke={MAIN}
        strokeWidth={s * 0.13}
        strokeLinecap="round"
      />
      <path
        d={`M ${cx + s * 0.26},${cy + s * 0.22} Q ${cx + s * 0.45},${cy + s * 0.15} ${cx + s * 0.41},${cy + s * 0.02}`}
        fill="none"
        stroke={MAIN}
        strokeWidth={s * 0.13}
        strokeLinecap="round"
      />
      {/* 갓 챙 */}
      <ellipse cx={cx} cy={cy - s * 0.22} rx={s * 0.33} ry={s * 0.075} fill={MAIN} />
      {/* 갓 모자 */}
      <path
        d={`M ${cx - s * 0.13},${cy - s * 0.22} Q ${cx - s * 0.16},${cy - s * 0.53} ${cx},${cy - s * 0.58} Q ${cx + s * 0.16},${cy - s * 0.53} ${cx + s * 0.13},${cy - s * 0.22} Z`}
        fill={MAIN}
      />
      {/* 갓 구름무늬 */}
      <circle cx={cx - s * 0.07} cy={cy - s * 0.4} r={s * 0.038} fill={PALE} opacity={0.9} />
      <circle cx={cx} cy={cy - s * 0.44} r={s * 0.046} fill={PALE} opacity={0.9} />
      <circle cx={cx + s * 0.07} cy={cy - s * 0.4} r={s * 0.038} fill={PALE} opacity={0.9} />
    </g>
  );
}

export default function MuseumBackground() {
  return (
    <div
      className="fixed inset-0 pointer-events-none select-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    >
      <svg
        viewBox="0 0 1440 1024"
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          {/* 창살 패턴 */}
          <pattern
            id="winLattice"
            x="0"
            y="0"
            width="22"
            height="22"
            patternUnits="userSpaceOnUse"
          >
            <rect width="22" height="22" fill="none" stroke={MAIN} strokeWidth="1.4" />
            <line x1="11" y1="0" x2="11" y2="22" stroke={MAIN} strokeWidth="0.7" />
            <line x1="0" y1="11" x2="22" y2="11" stroke={MAIN} strokeWidth="0.7" />
          </pattern>

          {/* 좌측 전시 조명 */}
          <radialGradient
            id="lightL"
            cx="155"
            cy="832"
            r="270"
            gradientUnits="userSpaceOnUse"
          >
            <stop offset="0%" stopColor={PALE} stopOpacity="0.82" />
            <stop offset="100%" stopColor={PALE} stopOpacity="0" />
          </radialGradient>

          {/* 우측 전시 조명 */}
          <radialGradient
            id="lightR"
            cx="1285"
            cy="832"
            r="270"
            gradientUnits="userSpaceOnUse"
          >
            <stop offset="0%" stopColor={PALE} stopOpacity="0.82" />
            <stop offset="100%" stopColor={PALE} stopOpacity="0" />
          </radialGradient>

          {/* 중앙 따뜻한 조명 */}
          <radialGradient id="centerGlow" cx="50%" cy="45%" r="40%">
            <stop offset="0%" stopColor="#EAD5B5" stopOpacity="0.45" />
            <stop offset="100%" stopColor={BG} stopOpacity="0" />
          </radialGradient>

          {/* 바닥 그라디언트 */}
          <linearGradient id="floorGrad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={PALE} stopOpacity="0" />
            <stop offset="100%" stopColor={PALE} stopOpacity="0.65" />
          </linearGradient>

          {/* 좌측→중앙 페이드 오버레이 */}
          <linearGradient id="leftBlend" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor={BG} stopOpacity="0" />
            <stop offset="100%" stopColor={BG} stopOpacity="1" />
          </linearGradient>

          {/* 우측→중앙 페이드 오버레이 */}
          <linearGradient id="rightBlend" x1="1" x2="0" y1="0" y2="0">
            <stop offset="0%" stopColor={BG} stopOpacity="0" />
            <stop offset="100%" stopColor={BG} stopOpacity="1" />
          </linearGradient>
        </defs>

        {/* ── 기본 배경 ── */}
        <rect width="1440" height="1024" fill={BG} />

        {/* ── 중앙 따뜻한 분위기 조명 ── */}
        <ellipse cx="720" cy="460" rx="540" ry="480" fill="url(#centerGlow)" />

        {/* ════════ 천장 ════════ */}
        <rect x="0" y="0" width="1440" height="80" fill={MAIN} opacity={0.3} />
        {/* 단청 느낌 띠 */}
        <rect x="0" y="76" width="1440" height="12" fill={LIGHT} opacity={0.6} />
        <rect x="0" y="86" width="1440" height="5" fill={MAIN} opacity={0.22} />
        {/* 처마 곡선 */}
        <path
          d="M -60,0 C 60,46 200,74 390,68"
          fill="none"
          stroke={MAIN}
          strokeWidth={7}
          opacity={0.26}
          strokeLinecap="round"
        />
        <path
          d="M 1500,0 C 1380,46 1240,74 1050,68"
          fill="none"
          stroke={MAIN}
          strokeWidth={7}
          opacity={0.26}
          strokeLinecap="round"
        />

        {/* ════════ 바닥 ════════ */}
        <rect x="0" y="836" width="1440" height="188" fill={PALE} opacity={0.55} />
        <rect x="0" y="836" width="1440" height="188" fill="url(#floorGrad)" />
        <line x1="0" y1="836" x2="1440" y2="836" stroke={LIGHT} strokeWidth={2.5} opacity={0.52} />
        {/* 투시 소실선 */}
        <line x1="0" y1="1024" x2="720" y2="680" stroke={LIGHT} strokeWidth={1.3} opacity={0.16} />
        <line x1="300" y1="1024" x2="720" y2="680" stroke={LIGHT} strokeWidth={0.7} opacity={0.1} />
        <line x1="1440" y1="1024" x2="720" y2="680" stroke={LIGHT} strokeWidth={1.3} opacity={0.16} />
        <line x1="1140" y1="1024" x2="720" y2="680" stroke={LIGHT} strokeWidth={0.7} opacity={0.1} />
        {/* 바닥 타일 가로선 */}
        <line x1="0" y1="874" x2="1440" y2="874" stroke={LIGHT} strokeWidth={0.9} opacity={0.22} />
        <line x1="0" y1="920" x2="1440" y2="920" stroke={LIGHT} strokeWidth={0.9} opacity={0.19} />
        <line x1="0" y1="974" x2="1440" y2="974" stroke={LIGHT} strokeWidth={0.9} opacity={0.16} />

        {/* ════════════ 좌측 패널 ════════════ */}

        {/* 좌측 벽 배경 */}
        <rect x="0" y="91" width="368" height="745" fill={PALE} opacity={0.2} />

        {/* 격자 창문 */}
        <g opacity={0.46}>
          {/* 창문 내부 */}
          <rect x="24" y="112" width="214" height="312" rx="5" fill={PALE} />
          {/* 창살 패턴 */}
          <rect x="24" y="112" width="214" height="312" rx="5" fill="url(#winLattice)" />
          {/* 아치 상단 채움 */}
          <path d="M 24,190 Q 131,108 238,190" fill={PALE} />
          {/* 아치 선 */}
          <path
            d="M 24,190 Q 131,108 238,190"
            fill="none"
            stroke={MAIN}
            strokeWidth={2.8}
          />
          {/* 외곽 프레임 */}
          <rect
            x="20"
            y="108"
            width="222"
            height="320"
            rx="7"
            fill="none"
            stroke={MAIN}
            strokeWidth={5.5}
          />
          {/* 창틀 하단 */}
          <rect x="16" y="424" width="230" height="15" rx={3} fill={MAIN} opacity={0.85} />
        </g>

        {/* 좌측 기둥 */}
        <g fill={MAIN} opacity={0.4}>
          {/* 공포 상단 */}
          <path d="M 283,56 L 293,92 L 391,92 L 401,56 Z" />
          {/* 공포 보 */}
          <rect x="289" y="90" width="98" height="19" rx={2} />
          {/* 주두 */}
          <rect x="303" y="107" width="70" height="21" rx={3} />
          {/* 기둥 몸통 */}
          <rect x="316" y="128" width="44" height="708" rx={9} />
          {/* 초석 */}
          <rect x="298" y="836" width="80" height="22" rx={4} />
        </g>

        {/* 좌측 전시 조명 광원 */}
        <ellipse cx="155" cy="836" rx="240" ry="145" fill="url(#lightL)" />

        {/* 전시 스포트라이트 콘 1 */}
        <path d="M 155,570 L 62,774 L 248,774 Z" fill={PALE} opacity={0.38} />
        {/* 좌대 1 — 달항아리 */}
        <Pedestal x={80} y={774} w={150} h={62} op={0.6} />
        <MoonJar cx={155} cy={722} size={94} op={0.7} />

        {/* 전시 스포트라이트 콘 2 */}
        <path d="M 275,632 L 238,812 L 312,812 Z" fill={PALE} opacity={0.28} />
        {/* 좌대 2 — 청자 */}
        <Pedestal x={232} y={812} w={90} h={42} op={0.5} />
        <TallVase cx={277} cy={764} size={76} op={0.62} />

        {/* ════════════ 우측 패널 ════════════ */}

        {/* 우측 벽 배경 */}
        <rect x="1072" y="91" width="368" height="745" fill={PALE} opacity={0.2} />

        {/* 격자 창문 */}
        <g opacity={0.46}>
          <rect x="1202" y="112" width="214" height="312" rx="5" fill={PALE} />
          <rect x="1202" y="112" width="214" height="312" rx="5" fill="url(#winLattice)" />
          <path d="M 1202,190 Q 1309,108 1416,190" fill={PALE} />
          <path
            d="M 1202,190 Q 1309,108 1416,190"
            fill="none"
            stroke={MAIN}
            strokeWidth={2.8}
          />
          <rect
            x="1198"
            y="108"
            width="222"
            height="320"
            rx="7"
            fill="none"
            stroke={MAIN}
            strokeWidth={5.5}
          />
          <rect x="1194" y="424" width="230" height="15" rx={3} fill={MAIN} opacity={0.85} />
        </g>

        {/* 우측 기둥 */}
        <g fill={MAIN} opacity={0.4}>
          <path d="M 1039,56 L 1049,92 L 1147,92 L 1157,56 Z" />
          <rect x="1053" y="90" width="98" height="19" rx={2} />
          <rect x="1067" y="107" width="70" height="21" rx={3} />
          <rect x="1080" y="128" width="44" height="708" rx={9} />
          <rect x="1062" y="836" width="80" height="22" rx={4} />
        </g>

        {/* 우측 전시 조명 광원 */}
        <ellipse cx="1285" cy="836" rx="240" ry="145" fill="url(#lightR)" />

        {/* 전시 스포트라이트 콘 3 */}
        <path d="M 1285,548 L 1192,754 L 1378,754 Z" fill={PALE} opacity={0.38} />
        {/* 좌대 3 — 불상 */}
        <Pedestal x={1210} y={754} w={150} h={82} op={0.6} />
        <Buddha cx={1285} cy={700} size={108} op={0.7} />

        {/* 전시 스포트라이트 콘 4 */}
        <path d="M 1163,640 L 1126,816 L 1200,816 Z" fill={PALE} opacity={0.28} />
        {/* 좌대 4 — 향로 */}
        <Pedestal x={1118} y={816} w={90} h={38} op={0.5} />
        <Incenser cx={1163} cy={784} size={64} op={0.62} />

        {/* ════════════ 마스코트 민속이 ════════════ */}
        <Minsoki cx={68} cy={926} s={100} op={0.8} />

        {/* ════════════ 중앙 UI 공간 확보 ════════════ */}
        {/* 좌측 패널→중앙 부드러운 전환 */}
        <rect x="296" y="0" width="148" height="1024" fill="url(#leftBlend)" />
        {/* 우측 패널→중앙 부드러운 전환 */}
        <rect x="996" y="0" width="148" height="1024" fill="url(#rightBlend)" />
        {/* 중앙 영역 맑은 처리 */}
        <rect x="444" y="0" width="552" height="1024" fill={BG} opacity={0.68} />
      </svg>
    </div>
  );
}
