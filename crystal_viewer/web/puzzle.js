import { StaticStructureView } from "/static/three_view.js";
import { PUZZLE_RING_COLORS } from "/static/colors.js";

// Puzzle mode (docs/PUZZLE_SPEC.md). Two quizzes share one flow — pick a quiz,
// pick a structure, answer — and both reuse the analysis-mode StaticStructureView
// so atoms, the unit cell and the (periodic-aware) animation match the analysis.
//   * "axis"      — a highlighted axis is shown; name its highest rotation fold.
//   * "operation" — one symmetry operation is animated; name its kind (+ fold).

let view = null;
let started = false;
let catalog = null;
let currentQuiz = "axis"; // "axis" | "operation" | "composition" | "mapping" | "point_group"
let currentOperationDifficulty = "normal"; // "normal" | "hard"
let currentKind = "molecule"; // structure picker toggle
let currentSourceKind = "molecule"; // kind of the loaded structure (for answer options)
let questions = [];
let currentQuestion = null;
let revealOperation = null; // operation index to animate for the reveal
let revealAnim = null; // active reveal animation token
let roundGeneration = 0; // invalidates async work from prior rounds/screens
let operationAnswered = false;
let mappingGuess = null; // mapping quiz: the atom the player clicked
let mappingRevealTarget = null; // mapping quiz: the correct target atom (after answering)
let compositionPlaybackGeneration = 0;
let compositionQuestionReady = false;
// Composition quiz: operations A and B preloaded so one continuous slider (A in
// [0, 0.5], B in [0.5, 1]) can scrub across both without resetting to 0 at the
// A/B boundary the way separately-loaded animations would.
let compositionSnapshots = { a: null, b: null };
let compositionActiveSnapshot = null;
let compositionLabelA = "";
let compositionLabelB = "";
let mappingPlaybackGeneration = 0;
let pendingOperationExample = null; // operation quiz: structure picked, awaiting difficulty
let pendingOperationLabel = "";
// Crystals default to unit-cell atoms only (periodic images off); the camera
// overlay's toggle flips this per structure, reset on every new startStructure().
let puzzleShowBoundaryImages = false;

const REVEAL_DURATION_MS = 1600;
const INFINITE = "inf";
const SHIFT_OPTIONS = ["1/6", "1/4", "1/3", "1/2", "2/3", "3/4", "5/6"];
// Mapping quiz: ring colours (source / player's guess / correct target). Colours
// come from colors.js so they cannot silently collide with the symmetry-element
// colours three_view.js draws in the same view (a blue guess ring used to sit
// almost on top of the axis colour before that module existed).
const PICK_SOURCE_COLOR = PUZZLE_RING_COLORS.source;
const PICK_GUESS_COLOR = PUZZLE_RING_COLORS.guess;
const PICK_TARGET_COLOR = PUZZLE_RING_COLORS.target;

// Operation-identify answer vocabulary (canonical kinds match game/operation_identify.py).
// `orders` lists the folds offered for that kind (null = kind only). The improper
// kind shown depends on the structure: molecules use rotoreflection Sn (回映),
// crystals use rotoinversion -n (回反).
const OP_KIND_MOLECULE = { kind: "rotoreflection", label: "回映", orders: [3, 4, 6] };
const OP_KIND_CRYSTAL = { kind: "rotoinversion", label: "回反", orders: [3, 4, 6] };

function operationKinds() {
  if (currentOperationDifficulty === "hard") {
    return [
      { kind: "screw", label: "らせん", orders: [2, 3, 4, 6] },
      { kind: "glide", label: "映進", orders: null },
    ];
  }
  // Linear molecules have a C∞ axis, so molecules offer ∞ as a rotation fold;
  // the crystallographic restriction means crystals never do.
  const rotationOrders = currentSourceKind === "crystal" ? [2, 3, 4, 6] : [2, 3, 4, 6, INFINITE];
  return [
    { kind: "rotation", label: "回転", orders: rotationOrders },
    { kind: "mirror", label: "鏡映", orders: null },
    { kind: "inversion", label: "反転", orders: null },
    currentSourceKind === "crystal" ? OP_KIND_CRYSTAL : OP_KIND_MOLECULE,
  ];
}

function el(id) {
  return document.getElementById(id);
}

function formatFormula(text) {
  // Render a chemical formula with subscripted digit runs (C6H6 -> C₆H₆).
  return String(text).replace(/(\d+)/g, "<sub>$1</sub>");
}

function formatOrder(order) {
  return order === INFINITE ? "無限回" : `${order}回`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatOperationNotation(notation) {
  return escapeHtml(notation).replace(/_([0-9]+)/g, "<sub>$1</sub>");
}

// Point-group symbols: Schoenflies (C3v) gets a subscript; HM/ITC (-3, m-3m) use
// the ASCII "-n" convention for a barred numeral (n̄), which browser_ui.js already
// renders as a real overline via the shared `.overline` CSS class (text-decoration:
// overline) — mirrored here since puzzle.js is a separate module and cannot import
// browser_ui.js's non-module formatGroupSymbol(). Without this, "-3" reads as
// "minus 3" instead of the crystallographic bar-3.
function formatPointGroupSymbol(symbol) {
  const text = escapeHtml(symbol);
  const schoenflies = text.match(/^([CDS])(\d+|∞)([a-z]*)$/);
  if (schoenflies) return `${schoenflies[1]}<sub>${schoenflies[2]}${schoenflies[3]}</sub>`;
  return text.replace(/-([0-9]+)/g, '<span class="overline">$1</span>');
}

function formatOperationAnswer(kind, order, shift = null, notation = null) {
  let text = kind;
  if (kind === "rotation") text = `${order}回回転`;
  if (kind === "mirror") text = "鏡映";
  if (kind === "inversion") text = "反転";
  if (kind === "rotoreflection") text = `回映（S${order}）`;
  if (kind === "rotoinversion") text = `回反（-${order}）`;
  if (kind === "screw") text = `${order}回らせん`;
  if (kind === "glide") text = "映進";
  if (shift) text += `、並進成分 ${shift}`;
  const escaped = escapeHtml(text);
  // The kind label already spells out Sn / -n, so repeating the notation would
  // read "回映（S3）（S3）". Compare loosely so a unicode minus or stray spacing
  // in the notation still counts as the same text.
  const canonical =
    kind === "rotoreflection" ? `S${order}` : kind === "rotoinversion" ? `-${order}` : null;
  const normalize = (value) => String(value).replace(/\s+/g, "").replace(/[−–—]/g, "-");
  const redundantNotation = canonical !== null && normalize(notation) === normalize(canonical);
  return notation && !redundantNotation
    ? `${escaped}（${formatOperationNotation(notation)}）`
    : escaped;
}

function formatOperationAnswers(answers) {
  // Coincident motions accept more than one name (e.g. CO2: 反転 or 鏡映).
  return (answers || [])
    .map((a) => formatOperationAnswer(a.kind, a.order, a.shift, a.notation || a.symbol))
    .join(" または ");
}

function displayLabel(example) {
  // display_formula is the formula as a chemist writes it (C6H6, not the
  // reduced HC); the analysis layer computes it, so there is no table here.
  return formatFormula(example.display_formula || example.formula || example.name);
}

// Which catalog count decides whether the current quiz can use a structure.
function puzzleCountKey() {
  if (currentQuiz === "operation") {
    return currentOperationDifficulty === "hard" ? "operation_hard" : "operation_normal";
  }
  return currentQuiz === "axis" ? "axis" : currentQuiz;
}

function isEligible(example) {
  // Structures carrying symmetry the answer vocabulary cannot name are analysis
  // only. This is not the same test as "has no questions": an icosahedral
  // cluster has plenty of C2/C3 questions, and asking only those would teach
  // that its 5-fold axes are not there.
  // The point-group quiz is the one exception: naming the whole structure's
  // point group (e.g. "Ih") never asks for a fold from the closed vocabulary,
  // so a 5-fold axis does not make that question a lie the way it would for
  // the other four (see beyond_quiz_vocabulary() in game/catalog.py).
  if (currentQuiz !== "point_group" && example.beyond_quiz_vocabulary) return false;
  // Counts come from the catalog, which the export step computes by running the
  // real question generators. Without them (filesystem fallback catalog) assume
  // playable rather than hiding everything.
  const counts = example.puzzle_counts;
  if (!counts) return true;
  return Number(counts[puzzleCountKey()] || 0) > 0;
}

function eligibleExamples(kind) {
  return (catalog?.[kind] || []).filter(isEligible);
}

// --- Screens: quiz select -> structure picker -> play ---

function invalidatePuzzleWork() {
  roundGeneration += 1;
  compositionPlaybackGeneration += 1;
  mappingPlaybackGeneration += 1;
  compositionQuestionReady = false;
  cancelReveal();
  operationAnswered = false;
  currentQuestion = null;
  if (view) {
    view.pathGeneration += 1;
    view.animationPaths.clear();
    view.setAnimationSourceAtoms(null);
    view.onAtomPick = null;
    view.clearPickMarkers();
    view.clearTargetMarkers();
    view.clearSymmetryElements();
    view.resetAnimation();
  }
  return roundGeneration;
}

// #puzzle-picker, #puzzle-quiz-select and the other pre-play screens are
// siblings of #puzzle-play, not descendants — when the fullscreen button
// (togglePuzzleFullscreen()) fullscreens #puzzle-play alone, the Fullscreen
// API only paints that element's own subtree, so switching to any of these
// sibling screens while fullscreen leaves the player looking at a screen the
// browser refuses to render. Every function below that leaves #puzzle-play
// exits fullscreen first so the normal (non-fullscreen) layout is what
// actually shows the destination screen.
function exitPuzzleFullscreenIfActive() {
  if (document.fullscreenElement === el("puzzle-play")) {
    document.exitFullscreen().catch(() => {});
  }
}

function showQuizSelect() {
  exitPuzzleFullscreenIfActive();
  invalidatePuzzleWork();
  el("puzzle-quiz-select").hidden = false;
  el("puzzle-operation-difficulty").hidden = true;
  el("puzzle-point-group-kind").hidden = true;
  el("puzzle-picker").hidden = true;
  el("puzzle-play").hidden = true;
  el("puzzle-back").hidden = false;
}

function showOperationDifficulty() {
  exitPuzzleFullscreenIfActive();
  invalidatePuzzleWork();
  el("puzzle-quiz-select").hidden = true;
  el("puzzle-point-group-kind").hidden = true;
  el("puzzle-operation-difficulty").hidden = false;
  el("puzzle-picker").hidden = true;
  el("puzzle-play").hidden = true;
  el("puzzle-back").hidden = false;
  // Molecules never score operation_hard (no screws/glides), but by the time this
  // screen shows a specific crystal has already been picked, so hide "hard" too
  // when that particular crystal has no hard questions of its own.
  const hardCard = document.querySelector('[data-operation-difficulty="hard"]');
  const counts = pendingOperationExample?.puzzle_counts;
  hardCard.hidden = counts ? Number(counts.operation_hard || 0) === 0 : false;
}

function showPointGroupKind() {
  exitPuzzleFullscreenIfActive();
  invalidatePuzzleWork();
  el("puzzle-quiz-select").hidden = true;
  el("puzzle-operation-difficulty").hidden = true;
  el("puzzle-point-group-kind").hidden = false;
  el("puzzle-picker").hidden = true;
  el("puzzle-play").hidden = true;
  el("puzzle-back").hidden = false;
}

function showPicker() {
  exitPuzzleFullscreenIfActive();
  invalidatePuzzleWork();
  el("puzzle-quiz-select").hidden = true;
  el("puzzle-operation-difficulty").hidden = true;
  el("puzzle-point-group-kind").hidden = true;
  el("puzzle-picker").hidden = false;
  el("puzzle-play").hidden = true;
  el("puzzle-back").hidden = false;
}

function showPlay() {
  el("puzzle-quiz-select").hidden = true;
  el("puzzle-operation-difficulty").hidden = true;
  el("puzzle-point-group-kind").hidden = true;
  el("puzzle-picker").hidden = true;
  el("puzzle-play").hidden = false;
  // Every screen keeps a single 戻る visible (see goBack()) so there is always
  // exactly one way back to wherever the player came from, never zero and
  // never two competing buttons with different targets (2026-09-09 follow-up).
  el("puzzle-back").hidden = false;
}

// Which screen led into the round currently showing on #puzzle-play, so
// goBack() can undo exactly that one step. Fixed by currentQuiz/currentSourceKind
// rather than a general navigation stack: the puzzle flow's shape is small and
// static (quiz-select is the only screen with more than one possible child), so
// hardcoding each screen's single parent is simpler than a generic history stack
// and just as correct.
function playParentScreen() {
  if (currentQuiz === "point_group") return "pointGroupKind";
  // Only crystals visit the difficulty screen (molecules skip straight from the
  // picker into play, see the picker's structure-button handler below).
  if (currentQuiz === "operation" && currentSourceKind === "crystal") return "operationDifficulty";
  return "picker";
}

// The single 戻る button's handler: undo exactly one step, to whichever screen
// led into the one currently showing. This replaces two earlier buttons that
// could show at once with conflicting targets (#puzzle-back, which always
// jumped straight to the hub, and #puzzle-picker-back, which stepped back to
// quiz-select only) — see docs/sessions/SESSION_REPORT_2026-09-09.md.
function goBack() {
  if (!el("puzzle-play").hidden) {
    const parent = playParentScreen();
    if (parent === "pointGroupKind") return showPointGroupKind();
    if (parent === "operationDifficulty") return showOperationDifficulty();
    return showPicker();
  }
  if (!el("puzzle-operation-difficulty").hidden) return showPicker();
  if (!el("puzzle-picker").hidden || !el("puzzle-point-group-kind").hidden) return showQuizSelect();
  window.setAppMode("select"); // already on quiz-select: one more step back is the hub
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.text()) || `${response.status} ${response.statusText}`);
  return response.json();
}

const STRUCTURE_KINDS = [
  { kind: "molecule", label: "分子" },
  { kind: "crystal", label: "結晶" },
];

async function buildPicker() {
  if (!catalog) catalog = await getJson("/api/examples");
  const root = el("puzzle-picker");
  root.innerHTML = "";
  // Offer a kind only when it has a playable structure for this quiz. This
  // derives what used to be hardcoded: molecules never score operation_hard
  // (no screws or glides) and crystals never score mapping, so the hard
  // operation quiz stays crystal-only and the mapping quiz molecule-only.
  const structureKinds = STRUCTURE_KINDS.filter((item) => eligibleExamples(item.kind).length > 0);
  if (!structureKinds.some((item) => item.kind === currentKind)) {
    currentKind = structureKinds[0]?.kind || "molecule";
  }

  const kindRow = document.createElement("div");
  kindRow.className = "puzzle-kind-row";
  for (const item of structureKinds) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `puzzle-kind${item.kind === currentKind ? " selected" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => {
      if (currentKind === item.kind) return;
      currentKind = item.kind;
      buildPicker();
    });
    kindRow.appendChild(button);
  }
  root.appendChild(kindRow);

  const heading = document.createElement("h2");
  heading.className = "puzzle-picker-title";
  heading.textContent = currentKind === "crystal" ? "結晶を選んでください" : "分子を選んでください";
  root.appendChild(heading);
  const list = document.createElement("div");
  list.className = "puzzle-picker-list";
  const examples = eligibleExamples(currentKind);
  // Formulae are not unique (diamond and graphite are both C), so disambiguate
  // with the structure name, but only where a label actually repeats.
  const labelCounts = new Map();
  for (const example of examples) {
    const label = displayLabel(example);
    labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
  }
  for (const example of examples) {
    const label = displayLabel(example);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "puzzle-structure";
    button.dataset.name = example.name;
    // Only the name/formula — the point/space group would give the answer away.
    const suffix =
      labelCounts.get(label) > 1 ? ` <span class="puzzle-structure-note">${escapeHtml(example.name)}</span>` : "";
    button.innerHTML = `<span class="puzzle-structure-name">${label}${suffix}</span>`;
    button.addEventListener("click", () => {
      // Screws/glides only exist in crystals, so the difficulty choice only makes
      // sense once a crystal has actually been picked; a molecule always plays
      // at normal difficulty and skips that screen entirely.
      if (currentQuiz === "operation" && example.kind === "crystal") {
        pendingOperationExample = example;
        pendingOperationLabel = label + suffix;
        showOperationDifficulty();
        return;
      }
      if (currentQuiz === "operation") currentOperationDifficulty = "normal";
      startStructure(example, label + suffix).catch(showError);
    });
    list.appendChild(button);
  }
  root.appendChild(list);
}

async function buildPointGroupKindScreen() {
  if (!catalog) catalog = await getJson("/api/examples");
  for (const button of document.querySelectorAll("[data-point-group-kind]")) {
    button.hidden = eligibleExamples(button.dataset.pointGroupKind).length === 0;
  }
}

async function startRandomPointGroupStructure(kind) {
  if (!catalog) catalog = await getJson("/api/examples");
  currentKind = kind;
  const examples = eligibleExamples(kind);
  if (!examples.length) throw new Error("この種類には出題できる構造がありません。");
  const example = examples[Math.floor(Math.random() * examples.length)];
  await startStructure(example);
}

function buildLegend() {
  const legend = el("puzzle-legend");
  if (!legend) return;
  legend.innerHTML = "";
  for (const [element, color] of view.legendItems || []) {
    const item = document.createElement("button");
    const visible = !view.hiddenElements?.has(element);
    item.type = "button";
    item.className = "puzzle-legend-item";
    item.dataset.visible = visible ? "true" : "false";
    item.title = `${element} の表示を切り替え`;
    item.innerHTML = `<span class="puzzle-legend-swatch" style="background:${color}"></span>${element}`;
    item.addEventListener("click", () => {
      const nextVisible = view.hiddenElements?.has(element);
      view.setElementVisibility(element, nextVisible);
      buildLegend();
    });
    legend.appendChild(item);
  }
}

function showError(error) {
  const result = el("puzzle-result");
  result.hidden = false;
  result.className = "puzzle-result miss";
  result.textContent = `エラー: ${error.message || error}`;
}

function showPrompt(message) {
  const result = el("puzzle-result");
  result.hidden = false;
  result.className = "puzzle-result miss";
  result.textContent = message;
}

async function startStructure(example, label = displayLabel(example)) {
  const generation = invalidatePuzzleWork();
  showPlay();
  // Name/formula only; the point/space group would reveal the answer.
  el("puzzle-structure-title").innerHTML = label;
  el("puzzle-question").textContent = "読み込み中…";
  el("puzzle-options").innerHTML = "";
  el("puzzle-result").hidden = true;
  el("puzzle-check").hidden = false;
  el("puzzle-check").disabled = true;
  el("puzzle-again").hidden = true;
  el("puzzle-open-analysis").hidden = true;
  el("puzzle-playback").hidden = true;
  el("puzzle-view-along").hidden = true;
  setStageCaption("");
  await getJson("/api/open_example", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind: example.kind, path: example.path, request_id: Date.now() }),
  });
  if (generation !== roundGeneration) return;
  // Render with the shared analysis-mode view (atoms + unit cell + fitted camera).
  // Crystals default to showing the periodic boundary atoms — a bare unit cell
  // can look incomplete or cut off — while molecules have no periodic images
  // at all, so the flag is moot for them either way. The camera overlay's
  // toggle (below) still lets the player turn it off per structure.
  puzzleShowBoundaryImages = example.kind === "crystal";
  view.showAnimationTargets = example.kind === "crystal";
  view.showAnimationTargetCopies = false;
  view.showTrajectories = false;
  el("puzzle-target-copies-toggle").hidden = example.kind !== "crystal";
  el("puzzle-boundary-images-toggle").hidden = example.kind !== "crystal";
  el("puzzle-trajectories").hidden = true;
  hidePuzzleGifButton();
  syncTargetCopiesButton();
  syncTrajectoriesButton();
  syncBoundaryImagesButton();
  applyBoundaryImagesQuery(example.kind);
  await view.refresh();
  if (generation !== roundGeneration) return;
  buildLegend();
  const endpoint =
    currentQuiz === "axis"
      ? `/api/puzzle/axis_orders${example.kind === "crystal" ? "?display_mode=source" : ""}`
      : currentQuiz === "composition"
        ? "/api/puzzle/composition"
        : currentQuiz === "mapping"
          ? "/api/puzzle/mapping"
          : currentQuiz === "point_group"
            ? "/api/puzzle/point_group"
            : `/api/puzzle/operations?difficulty=${currentOperationDifficulty}`;
  const payload = await getJson(endpoint);
  if (generation !== roundGeneration) return;
  currentSourceKind = payload.source_kind || example.kind;
  questions = payload.questions || [];
  if (!questions.length) {
    view.clearSymmetryElements();
    const noun = example.kind === "crystal" ? "結晶" : "分子";
    const what =
      currentQuiz === "axis"
        ? "回転軸"
        : currentQuiz === "composition"
          ? "2つの操作を合成してできる操作"
          : currentQuiz === "mapping"
            ? "移り先を答えられる原子"
            : currentQuiz === "point_group"
              ? "点群"
              : currentOperationDifficulty === "hard"
                ? "並進を含む操作（らせん・映進）"
                : "基本操作";
    el("puzzle-question").textContent = `この${noun}には出題できる${what}がありません。別の${noun}を選んでください。`;
    el("puzzle-options").innerHTML = "";
    el("puzzle-check").hidden = true;
    el("puzzle-again").hidden = true;
    el("puzzle-playback").hidden = true;
    return;
  }
  beginRound();
}

function pickQuestion() {
  // All four of these balance by answer type: a highly symmetric structure has
  // many equivalent operations (or axes, or products) of the common kinds, so
  // pick a random group (opaque answer-type bucket) first, then a random
  // question within it. Without this, benzene's axis quiz (two independent
  // "C2 only" axis classes vs one "C2/C3/C6" principal axis) drew "2回" twice
  // as often as the principal axis, purely from the raw axis count.
  if (
    currentQuiz === "operation"
    || currentQuiz === "composition"
    || currentQuiz === "mapping"
    || currentQuiz === "axis"
  ) {
    const groups = new Map();
    for (const question of questions) {
      const key = question.group ?? question.id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(question);
    }
    const keys = [...groups.keys()];
    const bucket = groups.get(keys[Math.floor(Math.random() * keys.length)]);
    return bucket[Math.floor(Math.random() * bucket.length)];
  }
  return questions[Math.floor(Math.random() * questions.length)];
}

function beginRound() {
  invalidatePuzzleWork();
  currentQuestion = pickQuestion();
  revealOperation = null;
  view.showTrajectories = false;
  view.updateTrajectoryLines();
  view.updateAnimationTargetMarkers();
  mappingGuess = null;
  mappingRevealTarget = null;
  const result = el("puzzle-result");
  result.hidden = true;
  el("puzzle-options").innerHTML = "";
  el("puzzle-check").hidden = false;
  el("puzzle-check").disabled = false;
  el("puzzle-again").hidden = true;
  el("puzzle-open-analysis").hidden = true;
  el("puzzle-slider").value = "0";
  el("puzzle-slider").hidden = false;
  setPuzzleSliderBreakpoints([]);
  el("puzzle-replay").disabled = false;
  el("puzzle-trajectories").hidden = true;
  const viewAlong = el("puzzle-view-along");
  viewAlong.hidden = !(currentQuiz === "axis" || currentQuiz === "mapping");
  viewAlong.textContent = currentQuiz === "mapping" ? "操作要素方向から見る" : "軸方向から見る";
  setStageCaption("");
  hidePuzzleGifButton();
  syncTrajectoriesButton();
  if (currentQuiz === "axis") beginAxisRound();
  else if (currentQuiz === "composition") beginCompositionRound();
  else if (currentQuiz === "mapping") beginMappingRound();
  else if (currentQuiz === "point_group") beginPointGroupRound();
  else beginOperationRound();
}

function beginAxisRound() {
  view.addAxis(
    { direction_cart: currentQuestion.direction_cart, point_cart: currentQuestion.point_cart },
    view.sceneSpan(),
  );
  view.render();
  el("puzzle-question").textContent = "青い軸は何回回転軸ですか？（回転だけを数え、回映やらせんは数えません）";
  const options = el("puzzle-options");
  const choices = [...currentQuestion.options];
  if (currentQuestion.infinite) choices.push(INFINITE);
  for (const order of choices) {
    const label = document.createElement("label");
    label.className = "puzzle-option";
    label.innerHTML = `<input type="radio" name="puzzle-order" value="${order}"><span>${formatOrder(order)}</span>`;
    options.appendChild(label);
  }
  el("puzzle-playback").hidden = true; // reveal controls appear after answering
}

// --- Point-group quiz (bare structure, no highlighted element -- name the whole
// thing's point group from a small multiple-choice list) ---

function beginPointGroupRound() {
  el("puzzle-question").textContent = "この構造の点群はどれですか？";
  const options = el("puzzle-options");
  for (const symbol of currentQuestion.options) {
    const label = document.createElement("label");
    label.className = "puzzle-option";
    label.innerHTML = `<input type="radio" name="puzzle-point-group" value="${escapeHtml(symbol)}"><span>${formatPointGroupSymbol(symbol)}</span>`;
    options.appendChild(label);
  }
  el("puzzle-playback").hidden = true; // nothing to animate for this quiz
}

function selectedPointGroup() {
  const checked = el("puzzle-options").querySelector('input[name="puzzle-point-group"]:checked');
  return checked ? checked.value : null;
}

async function onCheckPointGroup() {
  const generation = roundGeneration;
  const selected = selectedPointGroup();
  if (selected == null) {
    showPrompt("点群を1つ選んでください。");
    return;
  }
  el("puzzle-check").disabled = true;
  const result = await getJson("/api/puzzle/point_group/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_id: currentQuestion.id, selected }),
  });
  if (generation !== roundGeneration) return;
  const box = el("puzzle-result");
  box.hidden = false;
  box.className = `puzzle-result ${result.correct ? "hit" : "miss"}`;
  const verdict = result.correct ? "正解" : `不正解（正解は ${formatPointGroupSymbol(result.answer)}）`;
  const explanation = result.explanation ? `<div class="puzzle-explanation">${escapeHtml(result.explanation)}</div>` : "";
  box.innerHTML = `${verdict}${explanation}`;
  el("puzzle-again").hidden = false;
  el("puzzle-open-analysis").hidden = false;
}

// The structure this question asked about is already open server-side (from
// startStructure()'s /api/open_example call, still current — nothing here has
// re-opened a different one since), so switching to analysis mode just needs
// to show it. window.setAppMode is browser_ui.js's own top-level function,
// implicitly global as a classic (non-module) script.
function openCurrentStructureInAnalysis() {
  // The quiz round underneath is left exactly as it was (still showing on
  // #puzzle-play, just covered by the analysis screen), so fromPuzzle: true
  // relabels #analysis-back into a way straight back to it instead of the
  // usual hub reset — see resumePuzzleMode() in browser_ui.js.
  window.setAppMode("analysis", { fromPuzzle: true });
}

function beginOperationRound() {
  el("puzzle-question").textContent =
    currentOperationDifficulty === "hard"
      ? "このアニメーションで示された対称操作の種類と並進成分を選んでください。"
      : "このアニメーションはどの操作ですか？";
  renderOperationOptions();
  // The animation IS the question, so show the playback controls and play it now.
  revealOperation = currentQuestion.operation_index;
  el("puzzle-playback").hidden = false;
  setPuzzleGifButtonReady(false);
  playReveal().catch(showError);
}

function renderOperationOptions() {
  const root = el("puzzle-options");
  const kinds = operationKinds();
  const kindRow = document.createElement("div");
  kindRow.className = "puzzle-op-group";
  kindRow.innerHTML = `<span class="puzzle-op-label">操作の種類</span>`;
  for (const { kind, label } of kinds) {
    const item = document.createElement("label");
    item.className = "puzzle-option";
    item.innerHTML = `<input type="radio" name="op-kind" value="${kind}"><span>${label}</span>`;
    kindRow.appendChild(item);
  }
  root.appendChild(kindRow);

  // The order row is always present (with reserved height) so selecting a kind
  // that needs a fold does not resize/shift the layout; only its contents change.
  const orderRow = document.createElement("div");
  orderRow.className = "puzzle-op-group puzzle-op-order";
  root.appendChild(orderRow);

  const shiftRow = document.createElement("div");
  shiftRow.className = "puzzle-op-group puzzle-op-order";
  if (currentOperationDifficulty === "hard") {
    shiftRow.hidden = true;
    shiftRow.innerHTML = `<span class="puzzle-op-label">並進成分</span>`;
    for (const shift of SHIFT_OPTIONS) {
      const item = document.createElement("label");
      item.className = "puzzle-option";
      item.innerHTML = `<input type="radio" name="op-shift" value="${shift}"><span>${shift}</span>`;
      shiftRow.appendChild(item);
    }
    root.appendChild(shiftRow);
  }

  kindRow.addEventListener("change", () => {
    const config = kinds.find((k) => k.kind === kindRow.querySelector("input:checked")?.value);
    orderRow.innerHTML = config?.orders ? `<span class="puzzle-op-label">回転次数</span>` : "";
    for (const order of config?.orders || []) {
      const item = document.createElement("label");
      item.className = "puzzle-option";
      item.innerHTML = `<input type="radio" name="op-order" value="${order}"><span>${formatOrder(order)}</span>`;
      orderRow.appendChild(item);
    }
    if (currentOperationDifficulty === "hard") {
      shiftRow.hidden = !(config?.kind === "screw" || config?.kind === "glide");
    }
  });
}

// --- Composition quiz ("A then B" -> which single operation?) ---

function setStageCaption(text) {
  const caption = el("puzzle-stage-caption");
  if (caption) caption.textContent = text || "";
}

function shortOperationLabel(answer) {
  const notation = answer?.notation || answer?.symbol;
  if (notation) return String(notation).replace(/sigma/gi, "σ").replace(/_/g, "");
  if (answer?.kind === "rotation") return `C${answer.order}`;
  if (answer?.kind === "mirror") return "σ";
  if (answer?.kind === "inversion") return "i";
  if (answer?.kind === "rotoreflection") return `S${answer.order}`;
  if (answer?.kind === "rotoinversion") return `-${answer.order}`;
  return "?";
}

function compositionStageLabel(stage, answer) {
  return `${stage}（${shortOperationLabel(answer)}）`;
}

function beginCompositionRound() {
  // The product is always a point operation, so answer with the normal vocabulary.
  currentOperationDifficulty = "normal";
  el("puzzle-question").textContent =
    "操作A を行い、続けて操作B を行うと、全体としてどの1つの操作と同じになりますか？";
  renderOperationOptions();
  // The slider tracks whichever operation (A, then B, then — after checking —
  // the product) is currently loaded, the same way it does for the other
  // quizzes' reveal animations. Dragging it cuts the auto-play short (the same
  // cancelReveal() the input handler always calls), which hands control to the
  // player but also lets A's phase end early and roll straight into B.
  el("puzzle-playback").hidden = false;
  el("puzzle-check").disabled = true;
  hidePuzzleGifButton();
  setPuzzleSliderBreakpoints([0.5]);
  playCompositionQuestion().catch(showError);
}

async function playSingleOperation(operationIndex, generation, caption, isCurrent = null) {
  const stillCurrent = isCurrent || (() => generation === roundGeneration);
  if (!stillCurrent()) return false;
  view.setAnimationSourceAtoms(null);
  setStageCaption(caption);
  await view.loadAnimationPaths(
    Number(operationIndex),
    ++view.pathGeneration,
    view.showAnimationTargets ? "unit_cell" : "displayed",
  );
  if (!stillCurrent()) return false;
  view.resetAnimation();
  await animateReveal();
  return stillCurrent();
}

// Load one operation's paths without touching the shared reveal state, and hand
// back a snapshot of everything loadAnimationPaths sets on `view` so it can be
// swapped back in later — this is what lets A and B share one slider instead of
// each fully replacing `view.animationPaths` at reveal time.
async function fetchOperationSnapshot(operationIndex, generation) {
  await view.loadAnimationPaths(
    Number(operationIndex),
    generation,
    view.showAnimationTargets ? "unit_cell" : "displayed",
  );
  if (generation !== view.pathGeneration) return null;
  return {
    paths: view.animationPaths,
    boundary: view.boundaryContext,
    breakpoints: view.animationBreakpoints,
    maxTravel: view.maximumTravelDistance,
    durationSeconds: view.baseAnimationDurationSeconds,
    operationIndex: view.animationOperationIndex,
  };
}

function activateCompositionSnapshot(snapshot) {
  view.animationPaths = snapshot.paths;
  view.boundaryContext = snapshot.boundary;
  view.animationBreakpoints = snapshot.breakpoints;
  view.maximumTravelDistance = snapshot.maxTravel;
  view.baseAnimationDurationSeconds = snapshot.durationSeconds;
  view.animationOperationIndex = snapshot.operationIndex;
  view.updateAnimationTargetMarkers();
  view.updateTrajectoryLines();
}

// fraction in [0, 0.5] scrubs A (progress = fraction*2); [0.5, 1] scrubs B.
function applyCombinedCompositionProgress(fraction) {
  const clamped = Math.max(0, Math.min(fraction, 1));
  const inSecondHalf = clamped > 0.5;
  const snapshot = inSecondHalf ? compositionSnapshots.b : compositionSnapshots.a;
  if (!snapshot) return; // still loading — nothing to show yet
  if (compositionActiveSnapshot !== snapshot) {
    activateCompositionSnapshot(snapshot);
    compositionActiveSnapshot = snapshot;
  }
  const local = inSecondHalf ? (clamped - 0.5) * 2 : clamped * 2;
  view.setAnimationProgress(Math.min(local, 1));
}

function compositionCaptionForFraction(fraction) {
  return fraction <= 0.5 ? compositionLabelA : compositionLabelB;
}

async function playCompositionQuestion() {
  const generation = roundGeneration;
  const question = currentQuestion;
  const playback = ++compositionPlaybackGeneration;
  const isCurrent = () =>
    generation === roundGeneration
    && playback === compositionPlaybackGeneration
    && question === currentQuestion
    && !operationAnswered;
  compositionQuestionReady = false;
  compositionSnapshots = { a: null, b: null };
  compositionActiveSnapshot = null;
  el("puzzle-check").disabled = true;
  el("puzzle-replay").disabled = true;
  let completed = false;
  try {
    compositionLabelA = compositionStageLabel("操作A", question.operation_a);
    compositionLabelB = compositionStageLabel("操作B", question.operation_b);
    const gen = ++view.pathGeneration;
    setStageCaption(compositionLabelA);
    compositionSnapshots.a = await fetchOperationSnapshot(question.operation_index_a, gen);
    if (!isCurrent() || !compositionSnapshots.a) return;
    compositionSnapshots.b = await fetchOperationSnapshot(question.operation_index_b, gen);
    if (!isCurrent() || !compositionSnapshots.b) return;
    applyCombinedCompositionProgress(0);
    await animateProgress(0, 1, REVEAL_DURATION_MS * 2, (fraction) => {
      applyCombinedCompositionProgress(fraction);
      setStageCaption(compositionCaptionForFraction(fraction));
    });
    if (!isCurrent()) return;
    setStageCaption(`${compositionLabelA} → ${compositionLabelB}`);
    completed = true;
  } finally {
    if (isCurrent()) {
      compositionQuestionReady = completed;
      el("puzzle-check").disabled = !completed;
      el("puzzle-replay").disabled = false;
    }
  }
}

async function playCompositionProduct(generation = roundGeneration) {
  const playback = ++compositionPlaybackGeneration;
  const isCurrent = () =>
    generation === roundGeneration
    && playback === compositionPlaybackGeneration
    && operationAnswered
    && revealOperation != null;
  cancelReveal();
  el("puzzle-replay").disabled = true;
  setPuzzleSliderBreakpoints([]); // the product is a single 0→1 animation, not an A/B pair
  try {
    if (!await playSingleOperation(revealOperation, generation, "合成の結果", isCurrent)) return;
    if (!isCurrent()) return;
    setStageCaption("合成の結果");
    revealOperationElements(generation);
  } finally {
    if (isCurrent()) el("puzzle-replay").disabled = false;
  }
}

async function onCheckComposition() {
  const generation = roundGeneration;
  if (!compositionQuestionReady) {
    showPrompt("操作A・Bの再生が終わってから回答してください。");
    return;
  }
  const { kind, order } = selectedOperationAnswer();
  if (!kind) {
    showPrompt("操作の種類を選んでください。");
    return;
  }
  const config = operationKinds().find((k) => k.kind === kind);
  if (config?.orders && order == null) {
    showPrompt("回数も選んでください。");
    return;
  }
  compositionQuestionReady = false;
  compositionPlaybackGeneration += 1;
  cancelReveal();
  view.pathGeneration += 1;
  el("puzzle-check").disabled = true;
  el("puzzle-replay").disabled = true;
  let result;
  try {
    result = await getJson("/api/puzzle/composition/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: currentQuestion.id, kind, order }),
    });
  } catch (error) {
    // A transient POST failure must not strand the round with both controls
    // disabled.  Keep the already-viewed question answerable and replayable.
    if (generation === roundGeneration) {
      compositionQuestionReady = true;
      el("puzzle-check").disabled = false;
      el("puzzle-replay").disabled = false;
    }
    throw error;
  }
  if (generation !== roundGeneration) return;
  const box = el("puzzle-result");
  box.hidden = false;
  box.className = `puzzle-result ${result.correct ? "hit" : "miss"}`;
  const answerText = formatOperationAnswers(result.answers);
  box.innerHTML = result.correct ? `正解（${answerText}）` : `不正解（正解は ${answerText}）`;
  operationAnswered = true;
  el("puzzle-again").hidden = false;
  // Reveal the product: play its animation, then show its symmetry element.
  revealOperation = result.product_index;
  await playCompositionProduct(generation);
}

// --- Reveal animation (shared) ---

function setSlider(fraction) {
  el("puzzle-slider").value = String(Math.round(fraction * 1000));
}

// Mirrors analysis mode's #movement-progress <datalist> tick marks
// (setMovementBreakpoints() in browser_ui.js), scoped to #puzzle-slider.
// Composition is the only quiz whose slider spans more than one loaded
// animation (A in [0, 0.5], B in [0.5, 1]), so it is the only one with a
// breakpoint to show.
function setPuzzleSliderBreakpoints(values) {
  const list = el("puzzle-slider-stops");
  list.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = String(Math.round(value * 1000));
    list.appendChild(option);
  }
}

function animateProgress(from, to, durationMs, apply = (fraction) => view.setAnimationProgress(fraction)) {
  cancelReveal();
  return new Promise((resolve) => {
    const token = { resolve, start: performance.now() };
    revealAnim = token;
    const step = (now) => {
      if (revealAnim !== token) return; // superseded/cancelled
      const k = Math.min((now - token.start) / durationMs, 1);
      const fraction = from + (to - from) * k;
      apply(fraction);
      setSlider(fraction);
      if (k >= 1) {
        revealAnim = null;
        resolve();
        return;
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function animateReveal() {
  return animateProgress(0, 1, REVEAL_DURATION_MS);
}

function cancelReveal() {
  const anim = revealAnim;
  revealAnim = null;
  if (anim) anim.resolve();
}

async function playReveal() {
  if (revealOperation == null) return; // e.g. an infinite axis: nothing to play
  const generation = roundGeneration;
  el("puzzle-replay").disabled = true;
  if (!el("puzzle-save-gif").hidden) el("puzzle-save-gif").disabled = true;
  try {
    // Override the shared analysis selection so puzzle rounds animate their own
    // displayed/unit-cell atoms consistently.
    await view.loadAnimationPaths(Number(revealOperation), ++view.pathGeneration, view.showAnimationTargets ? "unit_cell" : "displayed");
    if (generation !== roundGeneration) return;
    view.resetAnimation();
    await animateReveal();
    revealOperationElements(generation);
  } finally {
    if (generation === roundGeneration) {
      el("puzzle-replay").disabled = false;
      if (!el("puzzle-save-gif").hidden) {
        el("puzzle-save-gif").disabled = !view.animationPaths.size;
      }
    }
  }
}

function revealOperationElements(generation = roundGeneration) {
  if (!operationAnswered || generation !== roundGeneration || revealOperation == null) return;
  // Use the current path generation. If a replay supersedes this request, the
  // StaticStructureView guard discards it, and replay completion calls us again.
  view.loadSymmetryElements(Number(revealOperation), view.pathGeneration).catch(() => {});
}

// --- Mapping quiz ("where does this atom go?") ---

function atomStartMarker(sourceAtom) {
  for (const instance of view.atomInstances.values()) {
    if (instance.sourceAtom === Number(sourceAtom)) {
      return { position: instance.start, radius: instance.radius };
    }
  }
  return null;
}

function refreshMappingMarkers({ revealTarget = null } = {}) {
  const entries = [{ atom: currentQuestion.source_atom_index, color: PICK_SOURCE_COLOR }];
  if (mappingGuess != null && mappingGuess !== currentQuestion.source_atom_index) {
    // The blue ring means "the destination I clicked", so keep it at that
    // original site while all atoms move during the reveal.  Following the atom
    // that happened to occupy the site would give the colour a different meaning.
    const guess = atomStartMarker(mappingGuess);
    if (guess) {
      entries.push({
        position: [...guess.position],
        color: PICK_GUESS_COLOR,
        // Sits inside the yellow ring, which the source atom carries at radius
        // 1.0. Leave the same visible gap the green answer ring has on the
        // outside, so a correct guess reads as three rings and not one band.
        radius: guess.radius * 0.84,
      });
    }
  }
  if (revealTarget != null) {
    // A fixed ring at the target atom's start position: the highlighted atom lands
    // inside it as the operation completes (the target atom itself moves away).
    const target = atomStartMarker(revealTarget);
    if (target) {
      entries.push({
        position: [...target.position],
        color: PICK_TARGET_COLOR,
        // A correct blue guess and the green answer are concentric.  Different
        // radii leave both visible instead of letting one ring completely hide.
        radius: target.radius * 1.18,
      });
    }
  }
  view.setPickMarkers(entries);
}

function orientPlanarMappingStructure() {
  const coords = (view.renderData?.atoms || [])
    .map((atom) => atom.cart?.map(Number))
    .filter((cart) => Array.isArray(cart) && cart.length === 3 && cart.every(Number.isFinite));
  if (coords.length < 2) return;
  const add = (a, b) => a.map((value, axis) => value + b[axis]);
  const subtract = (a, b) => a.map((value, axis) => value - b[axis]);
  const scale = (a, factor) => a.map((value) => value * factor);
  const dot = (a, b) => a.reduce((sum, value, axis) => sum + value * b[axis], 0);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const length = (a) => Math.sqrt(dot(a, a));
  const center = scale(coords.reduce(add, [0, 0, 0]), 1 / coords.length);
  const centered = coords.map((point) => subtract(point, center));
  const primary = centered.reduce(
    (best, point) => (length(point) > length(best) ? point : best),
    [0, 0, 0],
  );
  const extent = length(primary);
  if (!(extent > 1e-10)) return;

  // A cross product finds the normal regardless of how the molecule is oriented
  // in Cartesian space.  If all points are collinear, choose a stable direction
  // perpendicular to the molecular line instead.
  let normal = centered.reduce(
    (best, point) => (length(cross(primary, point)) > length(best) ? cross(primary, point) : best),
    [0, 0, 0],
  );
  if (length(normal) <= extent * extent * 1e-8) {
    const leastAligned = Math.abs(primary[0]) <= Math.abs(primary[1])
      && Math.abs(primary[0]) <= Math.abs(primary[2])
      ? [1, 0, 0]
      : Math.abs(primary[1]) <= Math.abs(primary[2]) ? [0, 1, 0] : [0, 0, 1];
    normal = cross(primary, leastAligned);
  }
  const normalLength = length(normal);
  if (!(normalLength > 1e-10)) return;
  normal = scale(normal, 1 / normalLength);
  const thickness = Math.max(...centered.map((point) => Math.abs(dot(point, normal))));
  if (thickness > extent * 0.08) return;
  view.viewAlongCartesianDirection(normal, center);
}

function beginMappingRound() {
  mappingGuess = null;
  const operationLabel = shortOperationLabel(currentQuestion.operation);
  el("puzzle-question").textContent =
    `黄色の原子は、操作 ${operationLabel} でどの原子に移りますか？`
    + "移り先の原子をクリックして回答してください。";
  setStageCaption("");
  el("puzzle-options").innerHTML = ""; // the answer is a click, not a radio choice
  el("puzzle-slider").hidden = true;
  el("puzzle-playback").hidden = true; // reveal controls appear after answering
  el("puzzle-check").disabled = false;
  hidePuzzleGifButton();
  // Face planar/linear molecules along their thin axis so candidate sites do not
  // overlap (notably benzene's C/H rings in the default oblique camera).
  orientPlanarMappingStructure();
  refreshMappingMarkers();
  // Show the operation's symmetry element (axis/plane/centre) up front, same as
  // the spec calls for — this is independent of the (now-removed) motion preview,
  // which only ever animated the atom, not this element.
  view.loadSymmetryElements(Number(currentQuestion.operation_index), view.pathGeneration).catch(() => {});
  view.onAtomPick = (atomIndex) => {
    if (operationAnswered || atomIndex == null) return;
    mappingGuess = Number(atomIndex);
    refreshMappingMarkers();
  };
}

async function playMappingReveal(generation = roundGeneration) {
  const playback = ++mappingPlaybackGeneration;
  const isCurrent = () =>
    generation === roundGeneration
    && playback === mappingPlaybackGeneration
    && operationAnswered;
  cancelReveal();
  el("puzzle-replay").disabled = true;
  view.setAnimationSourceAtoms(null);
  try {
    // The now-removed playMappingPreview() (2026-09-09) used to load this once
    // per round, before the player could even click 回答 — nothing does that
    // any more, so the reveal has to load its own animation path, the same way
    // playReveal() does for the other quizzes. Without this, view.animationPaths
    // stays empty (cleared by invalidatePuzzleWork() at round start) and 再生
    // animates nothing.
    await view.loadAnimationPaths(Number(currentQuestion.operation_index), ++view.pathGeneration, "displayed");
    if (!isCurrent()) return;
    view.resetAnimation();
    refreshMappingMarkers({ revealTarget: mappingRevealTarget });
    await animateProgress(0, 1, REVEAL_DURATION_MS);
  } finally {
    if (isCurrent()) el("puzzle-replay").disabled = false;
  }
}

async function onCheckMapping() {
  const generation = roundGeneration;
  if (mappingGuess == null) {
    showPrompt("移り先の原子をクリックしてください。");
    return;
  }
  mappingPlaybackGeneration += 1;
  cancelReveal();
  view.pathGeneration += 1;
  el("puzzle-check").disabled = true;
  let result;
  try {
    result = await getJson("/api/puzzle/mapping/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: currentQuestion.id, selected_atom_index: mappingGuess }),
    });
  } catch (error) {
    if (generation === roundGeneration) {
      el("puzzle-check").disabled = false;
    }
    throw error;
  }
  if (generation !== roundGeneration) return;
  operationAnswered = true;
  const box = el("puzzle-result");
  box.hidden = false;
  box.className = `puzzle-result ${result.correct ? "hit" : "miss"}`;
  box.textContent = result.correct ? "正解" : "不正解";
  el("puzzle-again").hidden = false;
  mappingRevealTarget = result.target_atom_index; // keep the guess ring; add the target
  // Reveal the full operation from rest: the yellow atom lands on the fixed green
  // destination ring while the rest of the molecule follows the same operation.
  el("puzzle-playback").hidden = false;
  await playMappingReveal(generation);
}

// --- Checking answers ---

async function onCheck() {
  if (!currentQuestion) return;
  if (currentQuiz === "axis") return onCheckAxis();
  if (currentQuiz === "composition") return onCheckComposition();
  if (currentQuiz === "mapping") return onCheckMapping();
  if (currentQuiz === "point_group") return onCheckPointGroup();
  return onCheckOperation();
}

function onReplay() {
  // Composition replays its own two-phase sequence (or the product, once answered).
  if (currentQuiz === "composition") {
    if (operationAnswered && revealOperation != null) {
      return playCompositionProduct();
    }
    return playCompositionQuestion();
  }
  if (currentQuiz === "mapping") {
    // #puzzle-playback (and its 再生 button) stays hidden until after answering,
    // so there is nothing to replay before then.
    if (!operationAnswered) return Promise.resolve();
    return playMappingReveal();
  }
  return playReveal();
}

function selectedOrder() {
  const checked = el("puzzle-options").querySelector('input[name="puzzle-order"]:checked');
  return checked ? checked.value : null;
}

async function onCheckAxis() {
  const generation = roundGeneration;
  const selected = selectedOrder();
  if (selected == null) {
    showPrompt("回数を1つ選んでください。");
    return;
  }
  el("puzzle-check").disabled = true;
  const result = await getJson("/api/puzzle/axis_orders/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_id: currentQuestion.id, selected_order: selected }),
  });
  if (generation !== roundGeneration) return;
  const box = el("puzzle-result");
  box.hidden = false;
  box.className = `puzzle-result ${result.correct ? "hit" : "miss"}`;
  box.textContent = result.correct ? "正解" : `不正解（正解は ${formatOrder(result.answer)}）`;
  revealOperation = result.reveal_operation;
  el("puzzle-again").hidden = false;
  if (revealOperation == null) {
    el("puzzle-playback").hidden = true; // infinite axis: nothing to animate
  } else {
    el("puzzle-playback").hidden = false;
    el("puzzle-trajectories").hidden = false;
    setPuzzleGifButtonReady(false);
    playReveal().catch(showError);
  }
}

function selectedOperationAnswer() {
  const root = el("puzzle-options");
  const kind = root.querySelector('input[name="op-kind"]:checked')?.value || null;
  const orderInput = root.querySelector('input[name="op-order"]:checked');
  const shiftInput = root.querySelector('input[name="op-shift"]:checked');
  // Raw value ("2" or "inf"); the server normalises it.
  return {
    kind,
    order: orderInput ? orderInput.value : null,
    shift: shiftInput ? shiftInput.value : null,
  };
}

async function onCheckOperation() {
  const generation = roundGeneration;
  const { kind, order, shift } = selectedOperationAnswer();
  if (!kind) {
    showPrompt("操作の種類を選んでください。");
    return;
  }
  const config = operationKinds().find((k) => k.kind === kind);
  if (config?.orders && order == null) {
    showPrompt("回数も選んでください。");
    return;
  }
  if (currentOperationDifficulty === "hard" && shift == null) {
    showPrompt("並進成分も選んでください。");
    return;
  }
  el("puzzle-check").disabled = true;
  const result = await getJson("/api/puzzle/operations/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_id: currentQuestion.id, kind, order, shift, difficulty: currentOperationDifficulty }),
  });
  if (generation !== roundGeneration) return;
  const box = el("puzzle-result");
  box.hidden = false;
  box.className = `puzzle-result ${result.correct ? "hit" : "miss"}`;
  const answerText = formatOperationAnswers(result.answers);
  if (result.correct) {
    // When a motion carries more than one valid name, say so.
    box.innerHTML = result.answers.length > 1 ? `正解（${answerText} のいずれも正解）` : `正解（${answerText}）`;
  } else {
    box.innerHTML = `不正解（正解は ${answerText}）`;
  }
  if (result.answers.length > 1) {
    // Merged because the atoms move alike, not because the operations are equal.
    box.innerHTML += `<br>原子の動きは同じに見えますが、${answerText} は別々の対称操作です。`;
  }
  el("puzzle-again").hidden = false;
  el("puzzle-trajectories").hidden = false;
  operationAnswered = true;
  // Reveal where the operation's symmetry element sits (axis / plane / centre /
  // glide arrow) now that the answer is in — the same elements the analysis shows.
  revealOperationElements(generation);
}

// --- Camera controls (reuse StaticStructureView's own methods) ---

const CAMERA_DIRECTIONS = {
  "puzzle-cam-left": "left",
  "puzzle-cam-right": "right",
  "puzzle-cam-up": "up",
  "puzzle-cam-down": "down",
  "puzzle-cam-roll-left": "roll-left",
  "puzzle-cam-roll-right": "roll-right",
};

function toggleProjection() {
  const button = el("puzzle-projection");
  const next = button.dataset.projection === "perspective" ? "orthographic" : "perspective";
  view.setProjection(next);
  button.dataset.projection = next;
  button.textContent = next === "perspective" ? "透視投影" : "平行投影";
}

function syncTargetCopiesButton() {
  el("puzzle-target-copies").checked = view.showAnimationTargetCopies !== false;
}

// The checkbox's own click already flips el("puzzle-target-copies").checked
// before this "change" handler runs, so read it rather than negating state.
function toggleTargetCopies() {
  view.showAnimationTargetCopies = el("puzzle-target-copies").checked;
  view.updateAnimationTargetMarkers();
}

// Crystal-only: mirrors analysis mode's "周期境界セルの原子" (#include-boundary-images)
// toggle, including its checkbox + label styling. Puzzle mode has no
// server-side postState round trip for this, so it carries the flag as a
// query param on the view's own render_data/animation_path requests instead
// (same mechanism display_mode=source already uses here).
function applyBoundaryImagesQuery(kind) {
  const flag = puzzleShowBoundaryImages ? 1 : 0;
  view.renderDataQuery = kind === "crystal" ? `?display_mode=source&boundary_images=${flag}` : "";
  view.animationPathQuery = kind === "crystal" ? `&display_mode=source&boundary_images=${flag}` : "";
  view.symmetryElementQuery = kind === "crystal" ? "&display_mode=source" : "";
}

function syncBoundaryImagesButton() {
  el("puzzle-boundary-images").checked = puzzleShowBoundaryImages;
}

async function toggleBoundaryImages() {
  puzzleShowBoundaryImages = el("puzzle-boundary-images").checked;
  applyBoundaryImagesQuery(currentSourceKind);
  await view.refresh();
}

function syncTrajectoriesButton() {
  const button = el("puzzle-trajectories");
  const enabled = view.showTrajectories === true;
  button.dataset.enabled = enabled ? "true" : "false";
  button.textContent = enabled ? "軌道: 表示" : "軌道: 非表示";
}

function toggleTrajectories() {
  view.showTrajectories = view.showTrajectories !== true;
  view.updateTrajectoryLines();
  syncTrajectoriesButton();
}

// Fullscreen covers the whole play screen (view + question/options), not just
// the canvas, so the player can keep answering while it's enlarged.
function syncPuzzleFullscreenButton() {
  const button = el("puzzle-view-fullscreen");
  const active = document.fullscreenElement === el("puzzle-play");
  button.textContent = active ? "全画面を終了" : "全画面";
  button.setAttribute("aria-label", active ? "全画面表示を終了する" : "全画面表示にする");
}

async function togglePuzzleFullscreen() {
  if (document.fullscreenElement === el("puzzle-play")) {
    await document.exitFullscreen();
  } else {
    await el("puzzle-play").requestFullscreen();
  }
}

function hidePuzzleGifButton() {
  const button = el("puzzle-save-gif");
  button.hidden = true;
  button.disabled = true;
  button.textContent = "GIFを保存";
}

function setPuzzleGifButtonReady(ready) {
  const button = el("puzzle-save-gif");
  button.hidden = false;
  button.disabled = !ready;
  button.textContent = "GIFを保存";
}

async function savePuzzleGif() {
  const button = el("puzzle-save-gif");
  if (view.recording) return;
  const controls = [...el("puzzle-play").querySelectorAll("button, input")];
  const disabled = new Map(controls.map(control => [control, control.disabled]));
  for (const control of controls) control.disabled = true;
  button.disabled = true;
  button.textContent = "保存中...";
  try {
    await view.recordGif("puzzle-animation");
  } finally {
    for (const [control, wasDisabled] of disabled) control.disabled = wasDisabled;
    button.textContent = "GIFを保存";
  }
}

function setupCameraControls() {
  for (const [id, direction] of Object.entries(CAMERA_DIRECTIONS)) {
    el(id).addEventListener("click", () => {
      view.rotateCamera(direction, Number(el("puzzle-cam-angle").value) || 0);
    });
  }
  el("puzzle-view-along").addEventListener("click", () => {
    if (!currentQuestion) return;
    if (currentQuiz === "axis") {
      view.viewAlongCartesianDirection(currentQuestion.direction_cart, currentQuestion.point_cart);
    } else if (currentQuiz === "mapping") {
      view.viewAlongCurrentOperation();
    }
  });
}

function setupControls() {
  el("puzzle-projection").addEventListener("click", toggleProjection);
  el("puzzle-target-copies").addEventListener("change", toggleTargetCopies);
  el("puzzle-boundary-images").addEventListener("change", () => toggleBoundaryImages().catch(showError));
  el("puzzle-trajectories").addEventListener("click", toggleTrajectories);
  el("puzzle-view-fullscreen").addEventListener("click", () => togglePuzzleFullscreen().catch(showError));
  document.addEventListener("fullscreenchange", () => {
    syncPuzzleFullscreenButton();
    view?.resize();
  });
  setupCameraControls();
  el("puzzle-check").addEventListener("click", () => onCheck().catch(showError));
  el("puzzle-replay").addEventListener("click", () => onReplay().catch(showError));
  el("puzzle-save-gif").addEventListener("click", () => savePuzzleGif().catch(showError));
  el("puzzle-slider").addEventListener("input", (event) => {
    cancelReveal();
    const fraction = Number(event.target.value) / 1000;
    // Pre-answer composition scrubs the combined A/B timeline; everything else
    // (including composition's post-answer product reveal) is a single load.
    if (currentQuiz === "composition" && !operationAnswered) {
      applyCombinedCompositionProgress(fraction);
      setStageCaption(compositionCaptionForFraction(fraction));
    } else {
      view.setAnimationProgress(fraction);
    }
  });
  el("puzzle-again").addEventListener("click", () => {
    if (questions.length) beginRound();
  });
  el("puzzle-open-analysis").addEventListener("click", openCurrentStructureInAnalysis);
  el("puzzle-other").addEventListener("click", () => {
    // The point-group quiz never lets the player pick a structure by name
    // (that would let them recognise it and answer from memory instead of
    // looking at it), so there is no named list to show here. It used to send
    // "another structure" back to the molecule/crystal kind screen, but that
    // meant re-clicking the very kind already in play just to continue —
    // instead, re-roll a new random structure of the same kind in place.
    // Switching kind (or quiz) still works via 戻る, which showPlay() keeps
    // visible for this quiz precisely because it no longer visits a screen.
    if (currentQuiz === "point_group") {
      startRandomPointGroupStructure(currentKind).catch(showError);
    } else {
      showPicker();
    }
  });
  // A single 戻る button now handles every screen (see goBack()); browser_ui.js
  // used to own this button's click handler (a plain jump to the hub), but that
  // no longer fits now that it means "one step back" instead — see
  // docs/sessions/SESSION_REPORT_2026-09-09.md.
  el("puzzle-back").addEventListener("click", () => goBack());
  for (const card of document.querySelectorAll("[data-quiz]")) {
    card.addEventListener("click", () => {
      currentQuiz = card.dataset.quiz;
      if (currentQuiz === "operation") {
        // Structure first, difficulty second (only crystals ask): see the
        // picker's structure-button handler above.
        currentOperationDifficulty = "normal";
        buildPicker().then(showPicker).catch(showError);
      } else if (currentQuiz === "point_group") {
        buildPointGroupKindScreen().then(showPointGroupKind).catch(showError);
      } else {
        buildPicker().then(showPicker).catch(showError);
      }
    });
  }
  for (const card of document.querySelectorAll("[data-operation-difficulty]")) {
    card.addEventListener("click", () => {
      currentOperationDifficulty = card.dataset.operationDifficulty || "normal";
      if (!pendingOperationExample) return;
      startStructure(pendingOperationExample, pendingOperationLabel).catch(showError);
    });
  }
  for (const card of document.querySelectorAll("[data-point-group-kind]")) {
    card.addEventListener("click", () => {
      startRandomPointGroupStructure(card.dataset.pointGroupKind).catch(showError);
    });
  }
}

function enterPuzzle() {
  if (!started) {
    started = true;
    view = new StaticStructureView(el("puzzle-view"));
    view.setBackgroundMode("light");
    // Puzzle clicks are never analysis selections. The mapping round installs its
    // own hook; all other puzzle screens simply ignore atom clicks.
    view.disableAtomSelection = true;
    setupControls();
  } else if (view) {
    view.setActive(true);
    view.resize();
  }
  showQuizSelect();
}

window.addEventListener("symmetry-enter-puzzle", enterPuzzle);
// Leaving the puzzle hides this canvas but does not destroy it, so stop its
// frame loop; the analysis view resumes its own on the way in.
window.addEventListener("symmetry-exit-puzzle", () => view?.setActive(false));
// Dispatched by resumePuzzleMode() (browser_ui.js) when "クイズに戻る" is
// pressed from the analysis screen openCurrentStructureInAnalysis() jumped to
// — unlike symmetry-enter-puzzle, this does not call showQuizSelect(): the
// #puzzle-play screen and its round state were never torn down, so just
// reactivate the (already-built) view the same way switching back to any
// already-open screen does.
window.addEventListener("symmetry-resume-puzzle", () => {
  if (view) {
    view.setActive(true);
    view.resize();
  }
});

// browser_ui.js's global ArrowLeft/ArrowRight handler is not mode-aware: it
// dispatches this event on `window` whenever pressed, and three_view.js's
// StaticStructureView subscribes to it per-instance, so the puzzle view moves
// right along with the analysis one. But puzzle.js never listened for it, so
// #puzzle-slider was left showing the pre-keypress position. Sync it here —
// guarded by in-puzzle so a keypress taken while looking at analysis mode
// doesn't touch this (hidden, but still stale) slider.
window.addEventListener("symmetry-animation-progress", (event) => {
  if (!document.body.classList.contains("in-puzzle")) return;
  setSlider(Number(event.detail?.progress) || 0);
});

// In --mode puzzle the classic UI script can dispatch its entry event before this
// module has registered the listener. Recover from that ordering by inspecting the
// already-rendered screen; `started` keeps the path idempotent.
if (el("puzzle-mode")?.hidden === false) queueMicrotask(enterPuzzle);
